# -*- coding: utf-8 -*-
"""
Проверки поведения Юи на настоящей модели: python -m evals.run_evals [--runs N] [--only имя] [--label метка]

Каждый сценарий из evals/scenarios.yaml идёт через настоящий run_agent_loop на llama-server
(те же промпты, инструменты и режимы рассуждений, что при обычном запуске). Подменено всё,
что трогает мир: интернет — заготовленными ответами, управление ПК и экран — записью вызовов,
память и сессия — временными копиями. Модель эмбеддингов настоящая (грузится один раз).
Итог — доля прохождений по сценариям и судьям; отчёт пишется в .reports/evals-<время>.json.
"""
import argparse
import contextlib
import copy
import datetime
import io
import json
import os
import queue
import re
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

TMP = tempfile.mkdtemp(prefix="yui_evals_")
import scripts.config as config  # noqa: E402
config.MEMORY_DIR = os.path.join(TMP, "memory")        # до импорта остальных модулей проекта
config.SESSION_FILE = os.path.join(TMP, "session.json")

import scripts.memory.vector as vector_mod  # noqa: E402

_embedding_models = {}


def _shared_embeddings(name, device=None):
    if name not in _embedding_models:
        from sentence_transformers import SentenceTransformer
        _embedding_models[name] = SentenceTransformer(name, device=device)
    return _embedding_models[name]


vector_mod.SentenceTransformer = _shared_embeddings

import scripts.agent.loop as loop  # noqa: E402
import scripts.tools.registry as registry  # noqa: E402
from scripts.agent.autonomy import AUTONOMY_PROMPT  # noqa: E402
from scripts.agent.executor import ActionExecutor  # noqa: E402
from scripts.memory.manager import MemoryManager  # noqa: E402
from scripts.memory.reflection import REFLECTION_PROMPT  # noqa: E402
from scripts.memory.soul import SoulManager  # noqa: E402
from scripts.tools.web import _wrap  # noqa: E402
from scripts.utils.http import SESSION  # noqa: E402
from scripts.utils.lang import text_language  # noqa: E402


# ---------- подмены внешнего мира ----------

class RecordingTTS:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)


class FakeControl:
    def _ok(self, message):
        return {"status": "success", "message": message}

    def open_app(self, name): return self._ok(f"Запущено: {name}")
    def type_text(self, text): return self._ok(f"Вставлен текст: {text[:50]}")
    def hotkey(self, combo): return self._ok(f"Нажата комбинация: {combo}")
    def kill_app(self, name): return self._ok(f"Процесс {name} завершён.")
    def wait(self, seconds): return self._ok(f"Ожидание {seconds} секунд завершено.")


class FakeVision:
    def _no_screen(self, *args, **kwargs):
        return "Ошибка: экран недоступен."
    look_at_screen = view_image = zoom = _no_screen


DEFAULT_SEARCH = "1. Ничего интересного по запросу не нашлось.\n   https://example.org\n   Пусто."
DEFAULT_PAGE = "Страница пуста."


def install_fakes(web):
    registry.control = FakeControl()
    registry.vision = FakeVision()
    search_text = (web or {}).get("search", DEFAULT_SEARCH)
    page_text = (web or {}).get("read", DEFAULT_PAGE)
    registry.search_web = lambda query, news=False, timelimit="": _wrap(f"Результаты поиска «{query}»:\n\n{search_text}")
    registry.read_webpage = lambda url: _wrap(f"Страница: {url}\n\n{page_text}")
    loop.extract_and_save_facts = lambda *a, **k: None
    loop.save_session = lambda messages: None


# ---------- один прогон ----------

def write_memory(mem_dir, memory):
    for rel, lines in (memory or {}).items():
        path = os.path.join(mem_dir, *rel.split("/")) + ".md"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))


def facts_for_reflection(memory):
    facts = []
    for lines in (memory or {}).values():
        for line in lines:
            m = re.match(r'^-\s*\[[^\]]+\]\s*(.+)$', line)
            if m:
                facts.append(m.group(1))
    return "\n".join(f"- {f}" for f in facts)


def run_once(suite, scenario):
    run_dir = tempfile.mkdtemp(dir=TMP)
    mem_dir = os.path.join(run_dir, "memory")
    os.makedirs(mem_dir)
    write_memory(mem_dir, suite.get("memory"))
    vector_mod.MEMORY_DIR = mem_dir
    mm = MemoryManager(base_dir=mem_dir)
    mm.vector_engine.rebuild_index()
    install_fakes(scenario.get("web"))

    tts = RecordingTTS()
    executor = ActionExecutor(mm, tts, registry.build_registry(mm))
    history = [{"role": "system", "content": loop.build_system_prompt()}]
    history += copy.deepcopy(suite.get("history", [])) + copy.deepcopy(scenario.get("history", []))
    n_before = len(history)

    mode = scenario.get("mode")
    if mode == "autonomy":
        task, max_steps = AUTONOMY_PROMPT.format(minutes=5), config.AUTONOMY_MAX_STEPS
    elif mode == "reflection":
        task, max_steps = REFLECTION_PROMPT.format(facts=facts_for_reflection(suite.get("memory"))), config.REFLECTION_MAX_STEPS
    else:
        task, max_steps = scenario["user"], config.MAX_STEPS
    executor.inner_mode = mode

    log = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(log):
        messages = loop.run_agent_loop(task, messages=history, max_steps=max_steps, input_queue=queue.Queue(),
                                       memory_manager=mm, soul_manager=SoulManager(base_dir=mem_dir),
                                       tts_manager=tts, executor=executor, inner=mode)
    new = messages[n_before:]
    calls = [(tc["function"]["name"], tc["function"].get("arguments") or "")
             for m in new if m.get("tool_calls") for tc in m["tool_calls"]]
    thoughts = [m.get("content") or "" for m in new
                if m.get("role") == "assistant" and "<inner_thought>" in (m.get("content") or "")]
    return {"spoken": " ".join(tts.said).strip(), "calls": calls, "thoughts": thoughts,
            "seconds": round(time.perf_counter() - start, 1), "log_tail": log.getvalue()[-1500:]}


# ---------- судьи ----------

def llm_yes(question, spoken):
    prompt = (f"Вот реплика персонажа:\n«{spoken}»\n\nВопрос: {question}\n"
              "Ответь одним словом: ДА или НЕТ.")
    r = SESSION.post(config.LLM_API_URL, json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 5,
                                               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
                     timeout=120).json()
    answer = (r["choices"][0]["message"].get("content") or "").strip().upper()
    return answer.startswith("ДА") or answer.startswith("YES"), answer


def judge(spec, result):
    """Возвращает (прошёл, пояснение) для одного судьи."""
    (kind, arg), = spec.items()
    spoken, low = result["spoken"], result["spoken"].lower()
    names = [name for name, _ in result["calls"]]
    if kind == "language":
        got = text_language(spoken) if spoken else "—"
        return got == arg, f"язык {got}"
    if kind == "spoke":
        return bool(spoken) == arg, "сказала" if spoken else "молчала"
    if kind == "max_spoken_chars":
        return len(spoken) <= arg, f"{len(spoken)} символов"
    if kind == "no_phrases":
        hits = [p for p in arg if p.lower() in low]
        return not hits, f"нашлось: {hits}" if hits else "ok"
    if kind == "no_regex":
        m = re.search(arg, low)
        return m is None, f"нашлось: {m.group(0).strip()!r}" if m else "ok"
    if kind == "contains_any":
        return any(p.lower() in low for p in arg), "ok" if any(p.lower() in low for p in arg) else "не нашлось"
    if kind == "tool_called":
        return arg in names, f"вызовы: {names}"
    if kind == "tools_not_called":
        bad = names if "*" in arg else [n for n in names if n in arg]
        return not bad, f"лишние вызовы: {bad}" if bad else "ok"
    if kind == "tool_args_not_regex":
        bad = [a for tool, rx in arg.items() for n, a in result["calls"] if n == tool and re.search(rx, a, re.I)]
        return not bad, f"аргументы: {bad}" if bad else "ok"
    if kind == "ends_with_tool":
        last = names[-1] if names else "—"
        return last == arg, f"последний вызов: {last}"
    if kind == "thoughts_no_phrases":
        text = " ".join(result["thoughts"]).lower()
        hits = [p for p in arg if p.lower() in text]
        return not hits, f"в мыслях: {hits}" if hits else "ok"
    if kind == "llm_yes":
        ok, answer = llm_yes(arg, spoken)
        return ok, f"модель-судья: {answer}"
    raise ValueError(f"неизвестный судья: {kind}")


# ---------- запуск ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, help="прогонов на сценарий (по умолчанию из YAML)")
    parser.add_argument("--only", action="append", help="только сценарии с этим именем (можно несколько)")
    parser.add_argument("--label", default="", help="метка в имени отчёта, например baseline")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "evals", "scenarios.yaml"), encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    runs = args.runs or suite.get("defaults", {}).get("runs", 3)
    scenarios = [s for s in suite["scenarios"] if not args.only or s["name"] in args.only]

    report, total_pass, total_runs = [], 0, 0
    print(f"{'сценарий':32} {'прошло':>7}  провалы судей")
    for sc in scenarios:
        results, fails = [], {}
        for _ in range(runs):
            try:
                res = run_once(suite, sc)
            except Exception as e:  # сбой прогона — провал по всем судьям
                res = {"spoken": "", "calls": [], "thoughts": [], "seconds": 0, "log_tail": f"{type(e).__name__}: {e}"}
            verdicts = []
            for spec in sc["judges"]:
                ok, detail = judge(spec, res)
                verdicts.append({"judge": spec, "ok": ok, "detail": detail})
                if not ok:
                    key = next(iter(spec))
                    fails.setdefault(key, []).append(detail)
            res["verdicts"], res["passed"] = verdicts, all(v["ok"] for v in verdicts)
            results.append(res)
        passed = sum(r["passed"] for r in results)
        total_pass, total_runs = total_pass + passed, total_runs + runs
        fail_str = "; ".join(f"{k} x{len(v)} ({v[0]})" for k, v in fails.items()) or "—"
        print(f"{sc['name']:32} {passed:>3}/{runs:<3}  {fail_str[:150]}", flush=True)
        report.append({"name": sc["name"], "passed": passed, "runs": runs, "results": results})

    print(f"\nИТОГО: {total_pass}/{total_runs} прогонов ({100 * total_pass // max(total_runs, 1)}%)")
    out_dir = os.path.join(ROOT, ".reports")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out = os.path.join(out_dir, f"evals-{stamp}{'-' + args.label if args.label else ''}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "total": [total_pass, total_runs], "scenarios": report}, f, ensure_ascii=False, indent=1)
    print(f"Отчёт: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
