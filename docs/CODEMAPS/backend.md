<!-- Generated: 2026-09-24 | Files scanned: 31 | Token estimate: ~750 -->
# Agent Core & Tools

No HTTP API of its own; "routes" = LLM tool calls.

## Tool calls (`tools/registry.py`: TOOLS schema, `build_registry(mm)`)
```
open_app(path)            -> ComputerControl.open_app      tools/control.py
type(text)                -> ComputerControl.type_text
hotkey(keys)              -> ComputerControl.hotkey
kill_app(app_name)        -> ComputerControl.kill_app
wait(seconds)             -> ComputerControl.wait
search_memory(query)      -> MemoryManager.search_facts    memory/manager.py
save_memory(path,content,confidence) -> save_memory_handler -> MemoryManager.save_fact
search_web(query,news,timelimit)     -> web.search_web     tools/web.py (ddgs)
read_webpage(url)         -> web.read_webpage              (trafilatura, wrapped in <web_content>)
look_at_screen(all_screens)          -> VisionState.look_at_screen  tools/vision.py
view_image(path)          -> VisionState.view_image
zoom_image(x1,y1,x2,y2,from_full)    -> VisionState.zoom
stay_silent(reason)       -> "SILENCE:..."        ends turn without speech
speak_aloud(text)         -> executor: TTS        inner turns only (inner_tools())
task_complete(reason)     -> "TASK_COMPLETE:..."  ends turn
```
Image results `{message, image_url}` -> tool msg (text) + separate user msg (image).

## Key files
```
agent/loop.py      447  classify_request, run_agent_loop, __main__ wiring
agent/executor.py  252  ActionExecutor: execute_tool_calls, finalize_response,
                        streaming TTS (feed_tts_chunk/flush), speech_text()
agent/parser.py    172  StreamParser.feed_chunk/finalize -> (reply, reasoning, tool_calls, emotion)
agent/context.py   182  estimate_chars, compress_context, extract_and_save_facts, inject_dynamic_context
agent/prompt.py    113  SYSTEM_PROMPT (static), build_system_prompt, get_dynamic_state
agent/will.py       86  check_willingness(messages) -> (decision, reason); will_note()
agent/emotion.py   118  Emotion enum, EmotionBridge (websocket stub), extract_emotion
agent/autonomy.py  135  mark_activity/seconds_since_activity (pause since last talk with YUI),
                        AutonomyManager -> input_queue ("autonomy", AUTONOMY_PROMPT)
agent/session.py    53  save/load/clear_session -> sessions/latest.json (images stripped)
tools/registry.py  274  TOOLS schemas + build_registry
tools/vision.py    197  VisionState, strip_images, append_image_message, message_chars
tools/web.py        97  search_web, read_webpage
tools/control.py   114  ComputerControl (subprocess, keyboard, pyperclip)
```

## Prompt layout (KV-cache friendly)
```
[0] system: SYSTEM_PROMPT (entity_core, personality, inference_rules RULE1-9, response_format) — never changes
... history ...
[n] user: <injected_context>time, hw, memory status, soul patch, auto memory</injected_context> + text
(+ ephemeral, not stored) will note, SILENCE_NUDGE_MESSAGE (p=0.15)
```

## Request classes (`classify_request`)
deep: task keyword stems (`\bнапиш`, `найди`, ...) or >=5 words -> T=0.6, 2048 tok
fast: short (<=6 words) + fast keyword, or <5 words -> T=0.5, 1024 tok
