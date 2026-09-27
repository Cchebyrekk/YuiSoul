# -*- coding: utf-8 -*-
import json

from scripts.agent.parser import StreamParser


def feed(parser, *deltas):
    for d in deltas:
        parser.feed_chunk({"choices": [{"delta": d}]})
    return parser.finalize()


def test_plain_content_and_reasoning():
    reply, reasoning, calls, emotion = feed(StreamParser(), {"reasoning_content": "думаю"},
                                            {"content": "При"}, {"content": "вет!"})
    assert (reply, reasoning, calls, emotion) == ("Привет!", "думаю", [], None)


def test_native_tool_call_assembled_from_chunks():
    reply, _, calls, _ = feed(
        StreamParser(),
        {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "search_", "arguments": '{"query": '}}]},
        {"tool_calls": [{"index": 0, "function": {"name": "memory", "arguments": '"кот"}'}}]},
        {"tool_calls": [{"index": 1, "function": {"name": "без_аргументов"}}]},  # неполный — отбрасывается
    )
    assert calls == [{"id": "c1", "type": "function", "function": {"name": "search_memory", "arguments": '{"query": "кот"}'}}]


def test_thought_split_and_output_tags():
    reply, reasoning, _, _ = feed(StreamParser(), {"content": "мысли\n</thought>\n<output>Ответ</output>"})
    assert reply == "Ответ"
    assert "мысли" in reasoning


def test_xml_tool_call_fallback_without_duplicates():
    xml = '<tool_call>{"name": "save_memory", "arguments": {"path": "user/pets"}}</tool_call>'
    parser = StreamParser()
    parser.feed_chunk({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "n1", "function": {
        "name": "save_memory", "arguments": json.dumps({"path": "user/pets"})}}]}}]})
    reply, _, calls, _ = feed(parser, {"content": f"Запомню. {xml}"})
    assert reply == "Запомню."
    assert len(calls) == 1  # тот же вызов в XML не дублируется

    _, _, calls, _ = feed(StreamParser(), {"content": xml})
    assert calls[0]["id"] == "xml_0" and calls[0]["function"]["name"] == "save_memory"


def test_tool_call_written_as_text_becomes_real_call():
    # Замерено на модели: "search_web{...}" и "<task_complete> <reason>...</reason>" текстом
    tools = {"search_web", "task_complete", "save_memory"}
    reply, _, calls, _ = feed(StreamParser(tools), {"content": 'Давай посмотрим.\nsearch_web{"query": "Нолан"}'})
    assert reply == "Давай посмотрим."
    assert calls[0]["function"] == {"name": "search_web", "arguments": '{"query": "Нолан"}'}

    reply, _, calls, _ = feed(StreamParser(tools), {"content": "<task_complete> <reason>Мысль исчерпана.</reason>"})
    assert reply == "" and json.loads(calls[0]["function"]["arguments"]) == {"reason": "Мысль исчерпана."}

    reply, _, calls, _ = feed(StreamParser(tools), {"content": 'Формат ответа: config{"a": 1}'})
    assert calls == [] and "config" in reply            # незнакомое имя — обычный текст


def test_emotion_self_report_extracted_and_clamped():
    reply, _, _, emotion = feed(StreamParser(), {"content": "<emotion>любопытство, 1.7</emotion> Ого"})
    assert reply == "Ого" and emotion == ("любопытство", 1.0)
    _, _, _, emotion = feed(StreamParser(), {"content": "<emotion>joy</emotion>Да"})
    assert emotion == ("joy", 0.6)


def test_preamble_removed_and_reset():
    parser = StreamParser()
    reply, _, _, _ = feed(parser, {"content": "Thinking Process: blah\nНастоящий ответ"})
    assert reply == "Настоящий ответ"
    parser.reset()
    assert parser.content_buffer == "" and parser.tool_calls == []


def test_numbered_thinking_written_as_reply_is_reasoning():
    leak = "Here's a thinking process that leads to the suggested response:\n\n1.  **Analyze the Input:**\n    * blah"
    reply, reasoning, _, _ = feed(StreamParser(), {"content": leak})
    assert reply == "" and "Analyze the Input" in reasoning


def test_broken_and_repeated_tool_calls_are_sanitized():
    # замерено: зациклившаяся модель — task_complete с незакрытым JSON; в истории он ломал каждый запрос (500)
    from scripts.agent.parser import sanitize_tool_calls
    broken = '{"reason":"Рефлексия завершена: ...</result>\n</function>\n</tool_call>\n<tool_call>'
    calls = [{"id": "1", "function": {"name": "task_complete", "arguments": broken}},
             {"id": "2", "function": {"name": "task_complete", "arguments": broken}},
             {"id": "3", "function": {"name": "save_memory", "arguments": '{"path": "user/a", "content": "x"}'}}]
    clean = sanitize_tool_calls(calls)
    assert [(c["id"], c["function"]["arguments"]) for c in clean] == [
        ("1", "{}"), ("3", '{"path": "user/a", "content": "x"}')]


def test_session_heals_poisoned_history(tmp_path, monkeypatch):
    import json
    import scripts.agent.session as session_mod
    path = tmp_path / "latest.json"
    poisoned = [{"role": "system", "content": "s"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "1", "function": {"name": "task_complete", "arguments": '{"reason":"незакрыт'}}]}]
    path.write_text(json.dumps({"saved_at": "x", "messages": poisoned}, ensure_ascii=False), encoding="utf-8")
    loaded = session_mod.load_session(str(path))
    assert loaded[1]["tool_calls"][0]["function"]["arguments"] == "{}"
