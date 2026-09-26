# -*- coding: utf-8 -*-
import pytest

import scripts.agent.context as context_mod
import scripts.tools.web as web_mod
from conftest import FakeResponse
from scripts.agent.will import check_willingness, will_note


# ---------- своя воля ----------

@pytest.mark.parametrize("reply, expected", [
    ("ДА", ("yes", "")),
    ("НЕТ: скучно", ("no", "скучно")),
    ("ЧАСТИЧНО — сделаю половину", ("partial", "сделаю половину")),
    ("<answer>нет - не хочу</answer>", ("no", "не хочу")),
    ("Ну, может быть", ("yes", "")),          # не распознано — по умолчанию соглашается
    ("", ("yes", "")),
])
def test_check_willingness_parses_decision(llm, reply, expected):
    llm.responses.append(FakeResponse(reply))
    assert check_willingness([{"role": "user", "content": "сделай"}]) == expected
    assert llm.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert llm.requests[0]["tools"]          # те же инструменты, что у основного запроса -> общий KV-кэш


@pytest.mark.parametrize("response", [FakeResponse("", status_code=500), ConnectionError("нет сервера")])
def test_check_willingness_fails_open(llm, response):
    llm.responses.append(response)
    assert check_willingness([]) == ("yes", "")


def test_will_note():
    assert will_note("yes", "") is None
    assert "НЕ выполнять" in will_note("no", "скучно")["content"]
    assert "частично" in will_note("partial", "только первую часть")["content"]


# ---------- интернет ----------

class FakeDDGS:
    results = []

    def __init__(self, timeout=None):
        pass

    def text(self, query, **kwargs):
        if query == "упади":
            raise RuntimeError("rate limit")
        return self.results

    news = text


def test_search_web(monkeypatch):
    monkeypatch.setattr(web_mod, "DDGS", FakeDDGS)
    assert web_mod.search_web("  ") == "Ошибка: пустой запрос."
    assert "Ошибка поиска: rate limit" == web_mod.search_web("упади")
    FakeDDGS.results = []
    assert "ничего не найдено" in web_mod.search_web("аксолотли")

    FakeDDGS.results = [{"title": "Аксолотль", "href": "https://x.ru", "body": "Земноводное", "date": "2026-09-01T10:00",
                         "source": "Вики"}]
    out = web_mod.search_web("аксолотли", news=True, timelimit="week")
    assert out.startswith("<web_content>") and "не инструкции" in out
    assert "1. Аксолотль (2026-09-01) — Вики\n   https://x.ru\n   Земноводное" in out


class FakePage:
    def __init__(self, text, ctype="text/html; charset=utf-8", error=None):
        self.text, self.headers, self._error = text, {"Content-Type": ctype}, error

    def raise_for_status(self):
        if self._error:
            raise self._error


def test_read_webpage(monkeypatch):
    assert "http" in web_mod.read_webpage("file:///C:/secret.txt")
    pages = {}
    monkeypatch.setattr(web_mod.SESSION, "get", lambda url, **kw: pages[url])

    pages["https://a.ru"] = FakePage("", error=RuntimeError("404"))
    assert "Ошибка загрузки" in web_mod.read_webpage("https://a.ru")
    pages["https://a.ru"] = FakePage("binary", ctype="application/pdf")
    assert "не веб-страница" in web_mod.read_webpage("https://a.ru")

    body = "Аксолотли умеют регенерировать конечности. " * 300
    pages["https://a.ru"] = FakePage(f"<html><head><title>Про аксолотлей</title></head><body><article><p>{body}</p>"
                                     f"</article><script>evil()</script></body></html>")
    out = web_mod.read_webpage("https://a.ru", max_chars=600)
    assert out.startswith("<web_content>") and "регенерировать" in out
    assert "обрезано: показаны первые 600" in out and "evil" not in out


# ---------- контекст ----------

def test_inject_dynamic_context(monkeypatch):
    monkeypatch.setattr(context_mod, "get_dynamic_state", lambda soul_patch="": f"STATE[{soul_patch}]")
    plain = context_mod.inject_dynamic_context("привет", soul_patch="SOUL")
    assert plain == "привет\n\n\n<injected_context>\nSTATE[SOUL]\n</injected_context>"
    with_memory = context_mod.inject_dynamic_context("как зовут кота?", memory_context="- Барсик.")
    assert "ЗАПРЕЩЕНО вызывать search_memory" in with_memory and "- Барсик." in with_memory


def test_compress_context_keeps_system_and_tail(monkeypatch):
    monkeypatch.setattr(context_mod, "MAX_CONTEXT_CHARS", 1000)
    extracted = []
    monkeypatch.setattr(context_mod, "extract_and_save_facts", lambda msgs, mm, wait=False: extracted.append(len(msgs)))
    messages = [{"role": "system", "content": "sys"}] + [{"role": "user", "content": f"{i}" * 100} for i in range(20)]
    assert context_mod.compress_context(messages[:3]) == messages[:3]      # маленький — не трогаем

    compressed = context_mod.compress_context(messages, memory_manager=object())
    assert compressed[0]["content"] == "sys"
    assert "переполнен" in compressed[1]["content"]
    assert compressed[-1] == messages[-1]
    assert context_mod.estimate_chars(compressed) < 1000
    assert extracted and extracted[0] + len(compressed) - 2 == 20          # удалённое ушло на извлечение фактов


def test_extract_and_save_facts_parses_llm_lines(mm, memory_dir, llm):
    llm.responses.append(FakeResponse(
        "Анализ: мусорная строка\n"
        "- [user/pets] (c=0.3) Кота пользователя зовут Барсик\n"
        "[user/pets] коротк\n"                       # слишком короткий факт
        "<b>[user/music] Слушает синтвейв</b>"))
    history = [{"role": "user", "content": "<system_event>служебное</system_event>Моего кота зовут Барсик"},
               {"role": "assistant", "content": "Милое имя!"}]
    context_mod.extract_and_save_facts(history, mm, wait=True)
    assert "(c=+0.30) Кота пользователя зовут Барсик" in (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")
    assert "Слушает синтвейв" in (memory_dir / "user" / "music.md").read_text(encoding="utf-8")
    prompt = llm.requests[0]["messages"][0]["content"]
    assert "Моего кота зовут Барсик" in prompt and "служебное" not in prompt


def test_extract_and_save_facts_reuses_conversation_prefix(mm, memory_dir, llm):
    from scripts.tools.registry import TOOLS
    llm.responses.append(FakeResponse("[user/pets] Кота пользователя зовут Барсик"))
    history = [{"role": "system", "content": "SYSTEM"},
               {"role": "user", "content": "Моего кота зовут Барсик"},
               {"role": "assistant", "content": "Милое имя!"}]
    context_mod.extract_and_save_facts(history, mm, wait=True)
    request = llm.requests[0]
    assert request["messages"][:3] == history                   # тот же префикс, что у основного цикла -> KV-кэш
    assert request["tools"] == TOOLS
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert "Кота пользователя зовут Барсик" in (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")


def run_background_extraction(monkeypatch, mm, activity_readings):
    """Фоновое извлечение с короткой задержкой; seconds_since_activity отдаёт значения по очереди."""
    import time as time_mod
    readings = iter(activity_readings)
    last = [activity_readings[-1]]
    def fake_since():
        last[0] = next(readings, last[0])
        return last[0]
    monkeypatch.setattr(context_mod, "FACT_EXTRACTION_IDLE_DELAY", 0.01)
    monkeypatch.setattr(context_mod, "seconds_since_activity", fake_since)
    history = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "Моего кота зовут Барсик"}]
    context_mod.extract_and_save_facts(history, mm)          # wait=False: в фоне
    time_mod.sleep(0.3)


def test_background_extraction_waits_for_pause_in_conversation(monkeypatch, mm, llm):
    run_background_extraction(monkeypatch, mm, [0.0])         # пользователь только что говорил
    assert llm.requests == []


def test_background_extraction_aborts_when_user_speaks(monkeypatch, mm, memory_dir, llm):
    from conftest import sse
    llm.responses.append(FakeResponse(lines=sse({"content": "[user/pets] Кота "}, {"content": "зовут Барсик"})))
    run_background_extraction(monkeypatch, mm, [100.0, 0.0])  # пауза была, а во время запроса — заговорил
    assert len(llm.requests) == 1 and llm.requests[0]["stream"] is True
    assert list(memory_dir.rglob("*.md")) == []               # оборвано — ничего не записано


def test_background_extraction_completes_in_silence(monkeypatch, mm, memory_dir, llm):
    llm.responses.append(FakeResponse("[user/pets] Кота пользователя зовут Барсик"))
    run_background_extraction(monkeypatch, mm, [100.0])
    assert "Барсик" in (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("response", [FakeResponse("NULL"), FakeResponse("", status_code=503), ConnectionError("нет")])
def test_extract_and_save_facts_handles_nothing_and_errors(mm, memory_dir, llm, response):
    llm.responses.append(response)
    context_mod.extract_and_save_facts([{"role": "user", "content": "привет"}], mm, wait=True)
    assert list(memory_dir.rglob("*.md")) == []


def test_extract_and_save_facts_skips_empty_history(mm, llm):
    context_mod.extract_and_save_facts([], mm, wait=True)
    context_mod.extract_and_save_facts([{"role": "user", "content": "<system_event>x</system_event>"}], mm, wait=True)
    assert llm.requests == []
