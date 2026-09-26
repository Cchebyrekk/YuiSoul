<!-- Generated: 2026-09-24 | Files scanned: 31 | Token estimate: ~450 -->
# Dependencies

## External services (local)
```
llama-server  127.0.0.1:8080 /v1/chat/completions   all LLM calls (loop, will, context,
              autonomy, reflection) via utils/http.SESSION (shared requests.Session)
              build: llama_things/turboquant-new/build/bin (TurboQuant KV + MTP)
              model: models/Qwen3.6-35B-A3B-Uncensored-Genesis-MTP-APEX-Compact.gguf
              must run with --parallel 1 (MTP) -> agent_is_working guard
Emotion WS    ws://127.0.0.1:8765  avatar stub (EMOTION_EXTRACTION_ENABLED=False)
```

## External services (internet)
```
DuckDuckGo via ddgs         search_web (region ru-ru, 5 results, 15 s timeout)
arbitrary URLs              read_webpage (trafilatura, <= 6000 chars)
models.silero.ai            one-time Silero v3_1_ru download -> speech/silero_model.pt
HuggingFace                 intfloat/multilingual-e5-base (first run)
```

## Python libraries (requirements.txt)
```
LLM/HTTP     requests
Speech       faster-whisper (STT, small, cpu int8), sounddevice, numpy
             torch + soundfile (Silero TTS, 48 kHz), pygame (playback), emoji (demojize ru)
Memory       sentence-transformers, rank-bm25
OS control   keyboard, pyperclip
Vision       Pillow (screenshots, zoom)
Web          ddgs, trafilatura
Optional     websocket-client (EmotionBridge)
```

## Bundled binaries (gitignored)
```
tts/          piper.exe + ru_RU-ruslan-medium.onnx, espeak-ng (not referenced by code; Silero is active)
scripts/tests/tts_t.py: standalone OmniVoice experiment (not used by the app)
llama_things/ llama.cpp builds (mainline, bee_llama, turboquant*)
models/       GGUF models
```
