# -*- coding: utf-8 -*-
import json
import threading

import scripts.agent.executor as executor_mod
from scripts.agent.executor import ActionExecutor
from scripts.agent.subagent import ASK_MAX_STEPS, run_ask

TOOLS = [{"type": "function", "function": {"name": n, "parameters": {}}}
         for n in ("search_memory", "search_web", "read_webpage", "run_command", "stay_silent")]
BASE = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "как сделать шлем?"}]
PAGE = "ОЧЕНЬ ДЛИННАЯ СТРАНИЦА " * 500


class Resp:
    def __init__(self, message, status=200):
        self.status_code, self._message = status, message

    def json(self):
        return {"choices": [{"message": self._message}]}


def call(name, args, cid="a1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def scripted(*messages):
    requests = []

    def post(url, json=None, timeout=None):
        requests.append(json)
        return Resp(messages[len(requests) - 1])
    return post, requests


def test_ask_researches_and_returns_summary():
    ran = []
    registry = {"search_web": lambda **kw: ran.append(("web", kw)) or "1. Гайд https://ex.com/helm",
                "read_webpage": lambda **kw: ran.append(("page", kw)) or PAGE,
                "run_command": lambda **kw: ran.append(("cmd", kw)) or "ok"}
    post, requests = scripted(
        {"content": "", "tool_calls": [call("search_web", {"query": "шлем 3D печать"})]},
        {"content": "", "tool_calls": [call("read_webpage", {"url": "https://ex.com/helm"}, "a2"),
                                       call("run_command", {"command": "del x"}, "a3")]},
        {"content": "Измерь окружность головы, печатай частями (https://ex.com/helm)."})
    answer = run_ask("как сделать шлем на 3D-принтере", BASE, TOOLS, registry, post=post)
    assert answer == "Измерь окружность головы, печатай частями (https://ex.com/helm)."
    assert ran == [("web", {"query": "шлем 3D печать"}), ("page", {"url": "https://ex.com/helm"})]   # не run_command
    for r in requests:                                    # тот же префикс и инструменты — кэш llama-server
        assert r["messages"][:2] == BASE and r["tools"] == TOOLS
        assert r["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(requests[2]["messages"][-2]["content"]) <= 5000                     # страница урезана
    assert "Недоступно" in requests[2]["messages"][-1]["content"]


def test_ask_stops_after_max_steps_and_handles_errors():
    looping = [{"content": "", "tool_calls": [call("search_web", {"query": "x"})]}] * ASK_MAX_STEPS
    post, requests = scripted(*looping, {"content": "Нашла немного: ..."})
    assert run_ask("x", BASE, TOOLS, {"search_web": lambda **kw: "r"}, post=post) == "Нашла немного: ..."
    assert len(requests) == ASK_MAX_STEPS + 1 and "Хватит искать" in requests[-1]["messages"][-1]["content"]
    post, _ = scripted(None)
    assert "не смогла" in run_ask("x", BASE, TOOLS, {}, post=lambda url, json=None, timeout=None: Resp({}, 500))


def test_executor_keeps_only_summary_in_history(monkeypatch):
    monkeypatch.setattr(executor_mod.time, "sleep", lambda s: None)
    seen = {}

    def fake_ask(question, base, tools, registry):
        seen.update(question=question, base=list(base), tools=tools)
        return "Выжимка: печатать частями."
    monkeypatch.setattr(executor_mod, "run_ask", fake_ask)
    ex = ActionExecutor.__new__(ActionExecutor)
    ex.mm, ex.inner_mode, ex.antirepeat, ex.registry = None, None, None, {}
    ex.current_tools = TOOLS
    history = list(BASE)
    messages, _, _ = ex.execute_tool_calls([call("ask", {"question": "как печатать шлем"})], history, threading.Event())
    assert seen["base"] == BASE and seen["tools"] == TOOLS and seen["question"] == "как печатать шлем"
    assert messages[-1]["content"] == "<ask_result>\nВыжимка: печатать частями.\n</ask_result>"
    assert ex.web_content_seen                         # после интернета управление ПК в этом ходе закрыто
