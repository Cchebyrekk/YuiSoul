<!-- Generated: 2026-09-24 | Files scanned: 31 | Token estimate: ~700 -->
# Architecture

Single-process Python app: local voice AI companion "YUI" (Windows).
Entry: `python -m scripts.agent.loop` (`scripts/agent/loop.py:358`).
LLM: external `llama-server` (OpenAI-compatible, `127.0.0.1:8080`, `--parallel 1`).

## Threads
```
main          input_queue.get() -> run_agent_loop()
keyboard      input() -> input_queue ("text")
STTManager    mic -> faster-whisper -> input_queue ("voice", interrupted=agent_busy)
TTSManager    speak queue -> Silero -> pygame
Autonomy      no talk with YUI >= 180 s, every >= 600 s -> input_queue ("autonomy", prompt)
Reflection    no talk >= 300 s, every >= 1200 s -> input_queue ("reflection", facts prompt); sleep consolidation in-thread
main          "autonomy"/"reflection" -> run_agent_loop(inner=...) — inner turn:
                text = <inner_thought> (printed, not spoken), tools = inner_tools() + speak_aloud,
                ends on task_complete / 2 plain thoughts in a row / max steps / user input in queue
```
"Idle" = time since last user input / end of YUI's reply (`autonomy.mark_activity`), not keyboard/mouse.
`agent_is_working` / `tts_active_event` block background LLM calls while YUI replies or speaks.
Background calls use `enable_thinking: False` (otherwise reasoning eats the token budget).

## One user turn (`run_agent_loop`)
```
user text
  -> [parallel] SoulManager.generate_soul_patch + MemoryManager.get_auto_context
  -> inject_dynamic_context()  (time, hw, memory status, soul patch -> tail of user msg)
  -> messages[0] = static SYSTEM_PROMPT   (stable -> KV-cache reuse)
  -> will.check_willingness()  -> ephemeral YES/NO/PARTIAL note
  -> loop step 1..MAX_STEPS(10):
       compress_context() if > MAX_CONTEXT_CHARS
       classify_request() -> fast/deep temp + max_tokens
       POST /v1/chat/completions stream=True (+ ephemeral will/silence notes)
       StreamParser -> reply, reasoning, tool_calls, <emotion>
       tool_calls? -> ActionExecutor.execute_tool_calls -> next step
       else        -> finalize_response -> TTS, emotion, save_session, break
  -> extract_and_save_facts() -> memory   (also on compress_context, background)
```

## Modules
```
scripts/
  config.py        all constants (paths, LLM, TTS/STT, memory, feature flags)
  agent/           turn loop, prompt, context, parsing, execution, will, autonomy
  tools/           tool schemas + handlers (OS control, vision, web)
  memory/          fact store, vector+BM25 search, soul, reflection
  speech/          stt.py (faster-whisper), tts.py (Silero)
  utils/           http.py (shared requests.Session), lang.py (ru/en by script)
  ui/, game/       empty stubs
  tests/tts_t.py   manual TTS smoke script
```
See: backend.md (agent+tools), data.md (memory), dependencies.md.
