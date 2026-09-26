# -*- coding: utf-8 -*-
"""Основной цикл хода с поддельным стримом llama-server: обычный ход, внутренний ход, прерывание."""
import json
import queue

import pytest

import scripts.agent.executor as executor_mod
import scripts.agent.loop as loop
from conftest import FakeResponse, sse
from scripts.agent.executor import ActionExecutor
from scripts.memory.soul import SoulManager
from scripts.tools.registry import build_registry


class FakeTTS:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)


@pytest.fixture
def env(monkeypatch, mm, memory_dir, llm):
    monkeypatch.setattr(loop, "extract_and_save_facts", lambda *a, **k: None)
    monkeypatch.setattr(loop, "save_session", lambda messages: None)
    monkeypatch.setattr(loop, "load_session", lambda: None)
    monkeypatch.setattr(loop, "WILL_CHECK_ENABLED", False)
    monkeypatch.setattr(loop, "SILENCE_NUDGE_CHANCE", 0.0)
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(executor_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr("scripts.agent.context.get_dynamic_state", lambda **kw: "STATE")
    tts = FakeTTS()
    executor = ActionExecutor(mm, tts, build_registry(mm))

    def run(task, inner=None, messages=None, q=None, max_steps=6):
        executor.inner_mode = inner
        return loop.run_agent_loop(task, messages=messages, max_steps=max_steps, input_queue=q or queue.Queue(),
                                   memory_manager=mm, soul_manager=SoulManager(base_dir=str(memory_dir)),
                                   tts_manager=tts, executor=executor, inner=inner)
    return run, llm, tts


def tool_call(name, args, cid="c1"):
    return {"tool_calls": [{"index": 0, "id": cid, "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


@pytest.mark.parametrize("text, mode", [
    ("привет", "fast"),
    ("да, давай", "fast"),
    ("далее расскажи подробнее про свою любимую игру", "deep"),  # "да" внутри "далее" не считается
    ("напиши стих", "deep"),
    ("как дела у тебя сегодня вечером в целом", "deep"),
    ("<system_note>Пользователь перебил</system_note> ок", "fast"),
])
def test_classify_request(text, mode):
    assert loop.classify_request(text) == mode


def test_user_input_pending():
    q = queue.Queue()
    q.put(("autonomy", "мысль", {}))
    assert loop._user_input_pending(q) is False
    q.put(("voice", "привет", {}))
    assert loop._user_input_pending(q) is True


def test_normal_turn_speaks_reply_and_uses_all_tools(env):
    run, llm, tts = env
    llm.responses.append(FakeResponse(lines=sse({"content": "Привет! Как ты?"})))
    messages = run("привет")
    assert messages[0]["role"] == "system"
    assert messages[1]["content"].startswith("привет") and "STATE" in messages[1]["content"]
    assert messages[-1] == {"role": "assistant", "content": "Привет! Как ты?"}
    assert tts.said == ["Привет! Как ты?"]
    tools = {t["function"]["name"] for t in llm.requests[0]["tools"]}
    assert {"type", "hotkey", "stay_silent"} <= tools and "speak_aloud" not in tools


def test_inner_turn_thinks_uses_tools_and_stays_quiet(env, memory_dir):
    run, llm, tts = env
    llm.responses += [
        FakeResponse(lines=sse({"content": "Интересно, запомню это."},
                               tool_call("save_memory", {"path": "user/games", "content": "Любит Z.A.T.O."}))),
        FakeResponse(lines=sse(tool_call("speak_aloud", {"text": "Слушай, а ты доиграл Z.A.T.O.?"}, "c2"))),
        FakeResponse(lines=sse(tool_call("task_complete", {"reason": "мысль исчерпана"}, "c3"))),
    ]
    messages = run("<system_event>подумай</system_event>", inner="autonomy")

    assert "Любит Z.A.T.O." in (memory_dir / "user" / "games.md").read_text(encoding="utf-8")
    assert tts.said == ["Слушай, а ты доиграл Z.A.T.O.?"]           # вслух только speak_aloud
    thought = next(m for m in messages if m.get("tool_calls"))
    assert thought["content"] == "<inner_thought>Интересно, запомню это.</inner_thought>"
    tools = {t["function"]["name"] for t in llm.requests[0]["tools"]}
    assert "speak_aloud" in tools and not tools & {"type", "hotkey", "open_app", "kill_app", "wait", "stay_silent"}


def test_inner_turn_ends_after_two_plain_thoughts(env):
    run, llm, tts = env
    llm.responses += [FakeResponse(lines=sse({"content": "Первая мысль."})),
                      FakeResponse(lines=sse({"content": "<inner_thought>Всё, можно завершать.</inner_thought>"}))]
    messages = run("подумай", inner="reflection")
    assert len(llm.requests) == 2
    assert messages[-1]["content"] == "<inner_thought>Всё, можно завершать.</inner_thought>"   # без двойной обёртки
    assert llm.requests[1]["messages"][-1]["content"] == loop.INNER_CONTINUE_NOTE["content"]   # просили продолжить
    assert tts.said == []


def test_inner_turn_ends_on_empty_reply_after_action(env):
    run, llm, tts = env
    llm.responses += [FakeResponse(lines=sse(tool_call("speak_aloud", {"text": "Эй, ты тут?"}))),
                      FakeResponse(lines=sse({"reasoning_content": "всё"}))]           # пустой ответ
    messages = run("подумай", inner="autonomy")
    assert len(llm.requests) == 2 and tts.said == ["Эй, ты тут?"]
    assert "<inner_thought></inner_thought>" not in [m.get("content") for m in messages]


def test_text_tool_call_in_inner_turn_is_executed(env):
    run, llm, _ = env
    llm.responses.append(FakeResponse(lines=sse({"content": "<task_complete> <reason>Готово.</reason>"})))
    messages = run("подумай", inner="reflection")
    assert len(llm.requests) == 1                                  # task_complete сработал, без лишних шагов
    assert any(m.get("tool_calls") for m in messages)


def test_inner_turn_yields_to_user(env):
    run, llm, _ = env
    q = queue.Queue()
    q.put(("voice", "Юи, ты тут?", {}))
    run("подумай", inner="autonomy", q=q)
    assert llm.requests == []            # даже не начала думать
    assert q.qsize() == 1                # реплика осталась для обычного хода


def test_normal_turn_drops_queued_inner_turns_but_injects_user_speech(env):
    run, llm, _ = env
    q = queue.Queue()
    q.put(("autonomy", "запланированная мысль", {}))
    q.put(("voice", "и ещё вопрос", {}))
    llm.responses.append(FakeResponse(lines=sse({"content": "Отвечаю."})))
    messages = run("привет", q=q)
    injected = [m["content"] for m in messages if m["role"] == "user"]
    assert any("и ещё вопрос" in c for c in injected)
    assert not any("запланированная мысль" in c for c in injected)


def test_reasoning_only_for_tasks_and_inner_turns(env, monkeypatch):
    run, llm, _ = env
    monkeypatch.setattr(loop, "WILL_CHECK_ENABLED", True)
    will_calls = []
    monkeypatch.setattr(loop, "check_willingness", lambda msgs: will_calls.append(1) or ("yes", ""))

    llm.responses.append(FakeResponse(lines=sse({"content": "Норм."})))
    run("как у тебя настроение сегодня вечером, чем занималась")          # разговор
    assert llm.requests[-1]["chat_template_kwargs"] == {"enable_thinking": False} and will_calls == []

    llm.responses.append(FakeResponse(lines=sse({"content": "Ищу."})))
    run("найди погоду на завтра")                                           # задача
    assert llm.requests[-1]["chat_template_kwargs"] == {"enable_thinking": True} and will_calls == [1]

    llm.responses.append(FakeResponse(lines=sse(tool_call("task_complete", {"reason": "всё"}))))
    run("подумай", inner="autonomy")                                         # размышление наедине
    assert llm.requests[-1]["chat_template_kwargs"] == {"enable_thinking": True} and will_calls == [1]


def test_remember_request_forces_tool_call_on_first_step(env, memory_dir):
    """Регрессия: модель отвечала "Запомнила", не вызывая save_memory."""
    run, llm, _ = env
    llm.responses += [FakeResponse(lines=sse(tool_call("save_memory", {"path": "user/family", "content": "Сестру зовут Лена"}))),
                      FakeResponse(lines=sse({"content": "Запомнила."}))]
    run("Запомни: мою сестру зовут Лена")
    assert llm.requests[0]["tool_choice"] == "required"
    assert "tool_choice" not in llm.requests[1]                  # дальше модель отвечает свободно
    assert "Лена" in (memory_dir / "user" / "family.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("text, forced", [
    ("Открой блокнот, пожалуйста", True),
    ("найди погоду на завтра", True),
    ("please open steam", True),
    ("почему небо синее?", False),                              # задача, но инструмент не обязателен
    ("как дела?", False),
])
def test_action_requests_force_a_tool_call(env, text, forced):
    run, llm, _ = env
    llm.responses.append(FakeResponse(lines=sse({"content": "Ок."})))
    run(text, max_steps=1)
    assert ("tool_choice" in llm.requests[0]) is forced


def test_silence_nudge_only_on_first_step_of_plain_chat(env, monkeypatch):
    run, llm, _ = env
    monkeypatch.setattr(loop, "SILENCE_NUDGE_CHANCE", 1.0)
    nudge = loop.SILENCE_NUDGE_MESSAGE["content"]
    llm.responses += [FakeResponse(lines=sse(tool_call("search_memory", {"query": "кот"}))),
                      FakeResponse(lines=sse({"content": "Ок."}))]
    run("расскажи что-нибудь про моего кота, если помнишь его")
    contents = [[m["content"] for m in r["messages"]] for r in llm.requests]
    assert nudge in contents[0] and nudge not in contents[1]     # посреди хода подсказки нет

    llm.responses.append(FakeResponse(lines=sse({"content": "Ищу."})))
    run("найди погоду")
    assert nudge not in [m["content"] for m in llm.requests[-1]["messages"]]   # задачи — без неё


def test_language_note_is_the_last_message(env, monkeypatch):
    run, llm, _ = env
    monkeypatch.setattr(loop, "SILENCE_NUDGE_CHANCE", 1.0)
    llm.responses.append(FakeResponse(lines=sse({"content": "Hey."})))
    run("Hey Yui, how's it going?")
    assert llm.requests[0]["messages"][-1]["content"] == loop.ENGLISH_REPLY_NOTE["content"]


def test_llm_error_then_step_limit(env):
    run, llm, _ = env
    llm.responses += [FakeResponse("ошибка", status_code=500)] * 3
    messages = run("привет", max_steps=1)
    assert messages[-1]["role"] == "user"   # ответа нет, но цикл завершился штатно
