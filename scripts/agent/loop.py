# -*- coding: utf-8 -*-
"""
Основной цикл агента YUI.
Оркестрирует все компоненты: память, душу, TTS, STT, автономию, эмоции.
Запускает фоновые потоки, обрабатывает ввод из очереди (клавиатура + голос),
выполняет итерации агента, управляет сессией.
"""
import queue
import threading
import time
import requests
import json
import re

from scripts.config import (
    MAX_STEPS,
    MAX_RETRIES,
    LLM_API_URL,
    FAST_TEMPERATURE,
    FAST_MAX_TOKENS,
    DEEP_TEMPERATURE,
    DEEP_MAX_TOKENS,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_REPEAT_PENALTY,
    STOP_TOKENS,
    LLM_TIMEOUT,
    ENABLE_AUTONOMY,
    ENABLE_REFLECTION
)
from scripts.agent.session import save_session, load_session, clear_session
from scripts.agent.prompt import build_system_prompt
from scripts.agent.context import compress_context, extract_and_save_facts, inject_dynamic_context
from scripts.agent.parser import StreamParser
from scripts.agent.executor import ActionExecutor
from scripts.agent.autonomy import AutonomyManager
from scripts.agent.emotion import EmotionBridge
from scripts.memory.manager import MemoryManager
from scripts.memory.soul import SoulManager
from scripts.memory.reflection import ReflectionManager
from scripts.speech.stt import STTManager
from scripts.speech.tts import TTSManager
from scripts.tools.registry import TOOLS, build_registry


# Глобальные флаги и очереди (будут созданы в __main__)
agent_is_working = threading.Event()
tts_active_event = threading.Event()



def classify_request(user_input: str) -> str:
    """Возвращает 'fast' или 'deep' в зависимости от запроса."""
    fast_keywords = ['привет', 'здравствуй', 'пока', 'спасибо', 'как дела', 'ok', 'да', 'нет']
    deep_keywords =     deep_keywords = [
        'почему', 'как сделать', 'напиши код', 'объясни', 'проанализируй',
        'создай', 'найди', 'открой', 'запусти', 'установи',
        'настрой', 'проверь', 'сравни', 'рассчитай', 'переведи',
        'составь', 'опиши', 'разработай', 'напиши', 'отредактируй'
    ]
    input_lower = user_input.lower()
    if any(kw in input_lower for kw in deep_keywords):
        return 'deep'
    if any(kw in input_lower for kw in fast_keywords):
        return 'fast'
    # По умолчанию — fast, если запрос короткий (< 5 слов)
    if len(user_input.split()) < 5:
        return 'fast'
    return 'deep'  # для всего остального

def run_agent_loop(user_task: str, messages: list = None, max_steps: int = MAX_STEPS,
                   input_queue: queue.Queue = None,
                   memory_manager: MemoryManager = None,
                   soul_manager: SoulManager = None,
                   tts_manager: TTSManager = None,
                   emotion_bridge: EmotionBridge = None,
                   executor: ActionExecutor = None) -> list:
    """
    Основной цикл выполнения одной задачи пользователя.
    Принимает все зависимости через параметры, чтобы быть тестируемым.
    Возвращает обновлённый список messages.
    """
    if memory_manager is None:
        memory_manager = MemoryManager()
    if soul_manager is None:
        soul_manager = SoulManager()
    if tts_manager is None:
        tts_manager = TTSManager()
    if emotion_bridge is None:
        emotion_bridge = EmotionBridge()
    if executor is None:
        # Строим реестр и executor
        registry = build_registry(memory_manager)
        executor = ActionExecutor(memory_manager, tts_manager, registry)

    # 1. Генерируем системный промпт с учётом души
    soul_patch = soul_manager.generate_soul_patch()
    system_prompt = build_system_prompt(soul_patch=soul_patch)

    # 2. Автоматический контекст из памяти (RAG)
    auto_mem = memory_manager.get_auto_context(user_task)
    # Инжектируем динамическое состояние (время, железо) и память
    user_input_final = inject_dynamic_context(user_task, auto_mem)

    # 3. Загружаем или инициализируем историю
    if messages is None:
        existing_session = load_session()
        if existing_session:
            print("[SYSTEM] Обнаружена предыдущая сессия. Восстановление...")
            messages = existing_session
            messages[0]["content"] = system_prompt  # обновляем системный промпт
            messages.append({"role": "user", "content": user_input_final})
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input_final}
            ]
    else:
        messages[0]["content"] = system_prompt
        messages.append({"role": "user", "content": user_input_final})

    # 4. Основной цикл итераций
    try:
        for step in range(1, max_steps + 1):
            print(f"\n--- ИТЕРАЦИЯ {step} ---")

            # Сжатие контекста (если нужно)
            messages = compress_context(messages, memory_manager)

            # Проверка очереди на новые сообщения от пользователя (интеррапты)
            if input_queue is not None:
                injected_texts = []
                while not input_queue.empty():
                    try:
                        source, new_input, metadata = input_queue.get_nowait()
                        if new_input == "EXIT":
                            continue
                        injected_texts.append(new_input)
                    except queue.Empty:
                        break
                if injected_texts:
                    merged = " ".join(injected_texts)
                    print(f"\n[SYSTEM LIVE INJECT]: Пользователь добавил: {merged}")
                    messages.append({
                        "role": "user",
                        "content": f"<system_note>Пользователь произнёс это во время твоей работы. Немедленно учти это и скорректируй действия, если нужно.</system_note>\n{merged}"
                    })

            # Запрос к LLM с повторными попытками
            raw_reply = ""
            reasoning = ""
            tool_calls = []
            success = False

            for attempt in range(MAX_RETRIES):
                agent_is_working.set()
                # Определяем режим
                mode = classify_request(user_task)  # или user_input_final, но лучше использовать исходный запрос
                if mode == 'fast':
                    temperature = FAST_TEMPERATURE   # 0.1
                    max_tokens = FAST_MAX_TOKENS     # 256
                else:
                    temperature = DEEP_TEMPERATURE   # 0.6
                    max_tokens = DEEP_MAX_TOKENS     # 2048

                # В payload:
                payload = {
                    "messages": messages,
                    "tools": TOOLS,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_k": DEFAULT_TOP_K,
                    "top_p": DEFAULT_TOP_P,
                    "repeat_penalty": DEFAULT_REPEAT_PENALTY,
                    "stop": STOP_TOKENS,
                    "stream": True
                }

                try:
                    response = requests.post(LLM_API_URL, json=payload, stream=True, timeout=LLM_TIMEOUT)
                    if response.status_code != 200:
                        print(f"[ERROR] LLM вернул {response.status_code}: {response.text}")
                        agent_is_working.clear()
                        time.sleep(1)
                        continue

                    # Парсим стрим через StreamParser
                    parser = StreamParser()
                    executor.start_streaming_tts()
                    print("[LLM STREAM]: ", end="", flush=True)
                    for line in response.iter_lines():
                        if not line: continue
                        decoded = line.decode('utf-8')
                        if not decoded.startswith('data: '): continue
                        json_str = decoded[6:]
                        if json_str.strip() == '[DONE]': break
                        try:
                            chunk = json.loads(json_str)
                            parser.feed_chunk(chunk)
                            # Извлекаем текст для вывода в реальном времени
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                token = delta['content']
                                print(token, end="", flush=True)
                                executor.feed_tts_chunk(token)
                            # Если есть reasoning_content, выводим его отдельно (можно серым)
                            if 'reasoning_content' in delta and delta['reasoning_content']:
                                # Выводим серым цветом (ANSI)
                                print(f"\033[90m{delta['reasoning_content']}\033[0m", end="", flush=True)
                        except json.JSONDecodeError:
                            continue
                    print()  # перевод строки после стрима
                    executor.flush_tts_buffer()

                    final_reply, reasoning, tool_calls = parser.finalize()
                    raw_reply = final_reply
                    print(f"\n[LLM STREAM] Ответ получен.")
                    if reasoning:
                        print(f"[LLM REASONING]: {reasoning[:200]}...")
                    if tool_calls:
                        print(f"[TOOL CALLS]: {json.dumps(tool_calls, indent=2)}")

                    success = True
                    break

                except requests.exceptions.Timeout:
                    print(f"[WARN] Таймаут LLM, попытка {attempt+1}/{MAX_RETRIES}")
                except Exception as e:
                    print(f"[ERROR] Ошибка при запросе к LLM: {e}")
                finally:
                    agent_is_working.clear()
                    time.sleep(0.5)

            if not success:
                print("[SYSTEM] Не удалось получить ответ от LLM после всех попыток. Завершаем итерацию.")
                continue

            # Обработка: если есть tool_calls — выполняем
            if tool_calls:
                messages, task_complete = executor.execute_tool_calls(tool_calls, messages, agent_is_working)
                if task_complete:
                    # Сохраняем факты и сессию
                    extract_and_save_facts(messages[1:], memory_manager)
                    save_session(messages)
                    return messages
                # Иначе продолжаем цикл (следующая итерация)
                continue

            # Если tool_calls нет — финализируем ответ
            should_exit = executor.finalize_response(raw_reply, reasoning, messages, agent_is_working)
            if should_exit:
                # Пустой ответ — выходим
                extract_and_save_facts(messages[1:], memory_manager)
                save_session(messages)
                return messages

            # Если есть финальный ответ без инструментов — сохраняем и завершаем цикл
            if raw_reply and not tool_calls:
                # Отправляем эмоцию, если есть
                emotion, intensity = emotion_bridge.extract_emotion(raw_reply)
                if emotion != "neutral":
                    emotion_bridge.send_emotion(emotion, intensity)
                extract_and_save_facts(messages[1:], memory_manager)
                save_session(messages)
                return messages

        # Если цикл завершился по максимуму шагов
        print("[SYSTEM] Достигнут лимит шагов, завершаю.")
        extract_and_save_facts(messages[1:], memory_manager)
        save_session(messages)
        return messages

    except KeyboardInterrupt:
        print("\n[SYSTEM] Ручная остановка (Ctrl+C). Спасаю факты...")
        if len(messages) > 1:
            extract_and_save_facts(messages[1:], memory_manager, wait=True)
        save_session(messages)
        return messages
    except Exception as e:
        print(f"\n[SYSTEM] Фатальная ошибка: {e}. Спасаю факты...")
        if len(messages) > 1:
            extract_and_save_facts(messages[1:], memory_manager, wait=True)
        save_session(messages)
        return messages


# ==================== ТОЧКА ВХОДА (если запускаем напрямую) ====================
if __name__ == "__main__":
    # Инициализация компонентов
    memory_mgr = MemoryManager()
    soul_mgr = SoulManager()
    tts_mgr = TTSManager(tts_active_event=tts_active_event)
    emotion_br = EmotionBridge()
    registry = build_registry(memory_mgr)
    executor = ActionExecutor(memory_mgr, tts_mgr, registry)

    # Очередь ввода (текст/голос)
    input_queue = queue.Queue()

    # STT (микрофон)
    stt_mgr = STTManager(input_queue=input_queue, agent_busy_event=agent_is_working)
    stt_mgr.start()

    # Автономия (если включена)
    autonomy = None
    if ENABLE_AUTONOMY:
        autonomy = AutonomyManager(
            soul_manager=soul_mgr,
            emotion_bridge=emotion_br
        )
        autonomy.start()

    # Фоновая рефлексия / консолидация памяти RAG 2.0 (если включена)
    reflection = None
    if ENABLE_REFLECTION:
        reflection = ReflectionManager(memory_manager=memory_mgr)
        reflection.start()

    # Поток ввода с клавиатуры
    def keyboard_thread(q):
        while True:
            try:
                user_input = input()
                if user_input.strip():
                    q.put(("text", user_input, {"timestamp": time.time(), "interrupted": False}))
            except KeyboardInterrupt:
                print("\n[SYSTEM] Завершение работы YUI.")
                q.put(("system", "EXIT", {}))
                break
            except Exception:
                pass

    kb_thread = threading.Thread(target=keyboard_thread, args=(input_queue,), daemon=True)
    kb_thread.start()

    messages = None
    try:
        print("\n[YUI SYSTEM] Агент запущен. Пиши текст или говори в микрофон.")

        while True:
            source, user_input, metadata = input_queue.get()
            if user_input == "EXIT":
                break

            # Если это голосовой интеррапт – добавляем пометку
            if metadata.get("interrupted"):
                user_input = (
                    f"<system_note>Пользователь произнёс это, пока ты думала или говорила, либо добавил сразу после твоего ответа. "
                    f"Учитывай это при формировании ответа.</system_note>\n{user_input}"
                )

            print(f"\n[INPUT SOURCE: {source}] Выполняю задачу...")
            messages = run_agent_loop(
                user_task=user_input,
                messages=messages,
                input_queue=input_queue,
                memory_manager=memory_mgr,
                soul_manager=soul_mgr,
                tts_manager=tts_mgr,
                emotion_bridge=emotion_br,
                executor=executor
            )

    except Exception as e:
        print(f"\n[SYSTEM FATAL] Падение основного цикла: {e}")
    finally:
        if autonomy:
            autonomy.stop()
        if reflection:
            reflection.stop()
        stt_mgr.stop()
        tts_mgr.stop()
        if messages:
            save_session(messages)
        print("[SYSTEM] YUI завершена.")