<!-- Generated: 2026-09-24 | Files scanned: 31 | Token estimate: ~550 -->
# Data (no database; files on disk)

## Stores
```
memory/                         long-term memory (MEMORY_DIR, gitignored)
  <category>/<topic>.md         one fact per line, optional "(c=0.90) " confidence prefix
  user/*.md                     facts about the user (read by SoulManager)
  system/yui/yui_character.md   YUI's self-discovered traits   (via save_memory)
  system/yui/yui_preferences.md YUI's likes/dislikes           (via save_memory)
  system/pc/*.md                hardware facts
  reflections/reflection_*.md   ReflectionManager output, read by SoulManager
  vectors.npy + metadata.json   VectorSearchEngine index (doc_id -> embedding)
sessions/latest.json            {"saved_at", "messages"} chat history, base64 images stripped
debug_logs/                     runtime logs
```

## Fact confidence (`memory/manager.py`)
```
-1  refuted   -> dropped below FACT_CONFIDENCE_DROP_THRESHOLD (-0.5)
 0  theory    (default; LLM never writes 1 itself)
 1  confirmed -> anchor (>= 0.999), immutable during consolidation
```

## Flows
```
save_fact(path, content, conf)
  -> dedupe: vector cosine > 0.90 | Jaccard >= 0.4 within file
  -> append line to .md -> VectorSearchEngine.add_document
search_facts(query) = vector (e5-base, >= 0.75) + BM25 (cached index) + recency boost
get_auto_context(query) -> <= 3 files x 2 lines, <= 400 chars -> injected_context
extract_and_save_facts(history)  LLM extracts facts -> save_fact
  called: end of turn (loop.py:312-352) and on compress_context (dropped msgs, bg)
ReflectionManager (idle)
  reflection cycle         -> reflections/reflection_*.md
  sleep consolidation (every 3rd cycle, ENABLE_SLEEP_CONSOLIDATION=False)
     read_mutable_and_anchor_lines -> LLM merge/compress -> rewrite_mutable_lines
SoulManager.generate_soul_patch -> user/* + system/yui/yui_*.md + latest reflections
  -> <soul_dynamic_state>
```

## Key files
```
memory/manager.py     554  MemoryManager, parse_fact_line, confidence tags
memory/vector.py      185  VectorSearchEngine (SentenceTransformer, npy/json index)
memory/reflection.py  395  ReflectionManager, _run_sleep_consolidation_cycle
memory/soul.py        123  SoulManager
agent/session.py       53  session persistence
```
