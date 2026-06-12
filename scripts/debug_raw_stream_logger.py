# debug_raw_stream_logger.py
import requests
import json
import time
import os

# Конфигурация
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
OUTPUT_DIR = "debug_logs"

# Тестовые сценарии, чтобы простимулировать разные паттерны модели
TEST_PROMPTS = [
    {
        "name": "1_simple_thought",
        "description": "Простой вопрос, должна выдать мысли и текст",
        "messages": [
            {"role": "system", "content": "Ты YUI. Отвечай в тегах <thought> и <output>."},
            {"role": "user", "content": "Что ты думаешь о Ryzen 9 7950X?"}
        ],
        "tools": []  # Без инструментов
    },
    {
        "name": "2_tool_call_open_app",
        "description": "Просьба открыть приложение, должен сработать Function Calling",
        "messages": [
            {"role": "system", "content": "Ты YUI. Отвечай в тегах <thought> и <output>. У тебя есть инструменты."},
            {"role": "user", "content": "Открой Блокнот, пожалуйста."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "open_app",
                    "description": "Запуск приложения.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Имя приложения"}
                        },
                        "required": ["path"]
                    }
                }
            }
        ]
    },
    {
        "name": "3_complex_tool_call",
        "description": "Сложный запрос с памятью, возможны галлюцинации",
        "messages": [
            {"role": "system", "content": "Ты YUI. Отвечай в тегах <thought> и <output>. У тебя есть инструменты."},
            {"role": "user", "content": "Сохрани факт, что я люблю пиццу, и найди информацию о моем коте."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Сохранение факта.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Поиск по памяти.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    }
]

def log_raw_stream(prompt_config: dict):
    """Отправляет запрос и логирует сырые чанки"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(OUTPUT_DIR, f"{timestamp}_{prompt_config['name']}.jsonl")
    
    payload = {
        "messages": prompt_config["messages"],
        "tools": prompt_config["tools"],
        "max_tokens": 1024,
        "temperature": 0.4,
        "stream": True
    }
    
    print(f"\n{'='*60}")
    print(f"[*] Тест: {prompt_config['name']} ({prompt_config['description']})")
    print(f"[*] Лог: {log_filename}")
    print(f"{'='*60}")
    
    try:
        with requests.post(LLAMA_SERVER_URL, json=payload, stream=True, timeout=120.0) as response:
            if response.status_code != 200:
                print(f"[!] Ошибка: HTTP {response.status_code} - {response.text}")
                return
            
            with open(log_filename, "w", encoding="utf-8") as f:
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded_line = line.decode("utf-8")
                    if not decoded_line.startswith("data: "):
                        continue
                    json_str = decoded_line[6:]
                    if json_str.strip() == "[DONE]":
                        break
                    
                    # Пишем сырой JSON в файл (одна строка = один чанк)
                    f.write(json_str + "\n")
                    
                    # Парсим для красивого вывода в консоль
                    try:
                        chunk = json.loads(json_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "")  # Критически важное поле!
                        tool_calls = delta.get("tool_calls", [])
                        
                        if content:
                            print(content, end="", flush=True)
                        if reasoning:
                            # Выводим мысли фиолетовым цветом, чтобы отличать
                            print(f"\033[95m{reasoning}\033[0m", end="", flush=True)
                        if tool_calls:
                            print(f"\n[TOOL_CALL_CHUNK]: {json.dumps(tool_calls, indent=2)}", flush=True)
                            
                    except json.JSONDecodeError:
                        print(f"\n[JSON_DECODE_ERROR]: {json_str}")
                        
            print(f"\n[+] Лог сохранен: {log_filename}")
            
    except requests.exceptions.RequestException as e:
        print(f"[!] Ошибка соединения: {e}")

if __name__ == "__main__":
    print("[SYSTEM] Запуск диагностического логгера llama.cpp")
    print(f"[SYSTEM] Целевой сервер: {LLAMA_SERVER_URL}")
    
    for prompt in TEST_PROMPTS:
        log_raw_stream(prompt)
        time.sleep(2)  # Пауза между запросами, чтобы не перегрузить GPU
    
    print("\n[SYSTEM] Диагностика завершена. Смотри папку debug_logs/")