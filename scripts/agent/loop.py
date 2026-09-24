# -*- coding: utf-8 -*-
"""
Основной цикл агента YUI.
Оркестрирует все компоненты: память, душу, TTS, STT, автономию, эмоции.
Запускает фоновые потоки, обрабатывает ввод из очереди (клавиатура + голос),
выполняет итерации агента, управляет сессией.
"""
import concurrent.futures
import queue
import random
import threading
import time
import requests
import json
import re
from typing import Optional

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
    ENABLE_REFLECTION,
    SILENCE_NUDGE_CHANCE,
    WILL_CHECK_ENABLED,
    AUTONOMY_MAX_STEPS,
    AUTONOMY_ALLOW_PC_CONTROL,
    REFLECTION_MAX_STEPS
)
from scripts.utils.http import SESSION
from scripts.agent.session import save_session, load_session, clear_session
from scripts.agent.prompt import build_system_prompt
from scripts.agent.context import compress_context, extract_and_save_facts, inject_dynamic_context
from scripts.agent.parser import StreamParser
from scripts.agent.executor import ActionExecutor, speech_text
from scripts.agent.will import check_willingness, will_note
from scripts.agent.autonomy import AutonomyManager, mark_activity
from scripts.agent.emotion import EmotionBridge
from scripts.memory.manager import MemoryManager
from scripts.memory.soul import SoulManager
from scripts.memory.reflection import ReflectionManager
from scripts.speech.stt import STTManager
from scripts.speech.tts import TTSManager
from scripts.tools.registry import TOOLS, build_registry, inner_tools
from scripts.tools.vision import strip_images


# Глобальные флаги и очереди (будут созданы в __main__)
agent_is_working = threading.Event()
tts_active_event = threading.Event()

# "Право на тишину" (по образцу processIgnoreChance из kuni): эфемерная
# подсказка, добавляемая ТОЛЬКО в запрос к LLM (не сохраняется в messages),
# с вероятностью SILENCE_NUDGE_CHANCE. Напоминает, что можно промолчать.
SILENCE_NUDGE_MESSAGE = {
    "role": "user",
    "content": (
        "<system_note>Кстати, ты не обязана отвечать на это прямо сейчас. Если "
        "не хочется отвечать, ответ не нужен, или ты ещё не решила, что сказать — "
        "можешь просто позвать stay_silent и промолчать. Это нормально.</system_note>"
    ),
}



# Внутренний ход: после мысли без инструментов — эфемерная подсказка продолжить или закончить.
INNER_CONTINUE_NOTE = {
    "role": "user",
    "content": (
        "<system_note>Это была твоя мысль про себя. Если хочется — продолжай: развивай её, "
        "поищи что-то, запомни вывод, скажи что-то вслух через speak_aloud. Если мысль исчерпана — "
        "вызови task_complete.</system_note>"
    ),
}


# Ключевые слова ищутся только с начала слова (\b), а не как подстроки:
# раньше "да" находилось в "далее", "нет" — в "интернет", "пока" — в "покажи".
# Слова-задачи — это основы: \bнапиш совпадёт с "напиши", "напишешь", "написать" нет.
DEEP_KEYWORDS_RE = re.compile(r'\b(' + '|'.join([
    'почему', 'как сделать', 'объясни', 'проанализируй', 'разбер',
    'создай', 'найди', 'поищи', 'загугли', 'открой', 'запусти', 'установи', 'закрой',
    'настрой', 'проверь', 'сравни', 'рассчитай', 'посчитай', 'переведи',
    'составь', 'опиши', 'разработай', 'напиш', 'отредактируй', 'исправь', 'почини',
    'переименуй', 'удали', 'скачай', 'сделай', 'посмотри', 'глянь', 'прочитай', 'придумай',
]) + r')', re.IGNORECASE)

# Короткие реплики: целые слова/фразы (\b с обеих сторон).
FAST_KEYWORDS_RE = re.compile(r'\b(' + '|'.join([
    'привет', 'здравствуй', 'пока', 'спасибо', 'как дела', 'ок', 'ok', 'да', 'нет', 'угу', 'ага',
]) + r')\b', re.IGNORECASE)

FAST_MAX_WORDS = 6  # длиннее — уже не "просто реплика", даже если в ней есть "да" или "спасибо"


def classify_request(user_input: str) -> str:
    """Возвращает 'fast' или 'deep' в зависимости от запроса."""
    # Служебные пометки (например, про голосовой интеррапт) — не слова пользователя
    text = re.sub(r'<system_note>.*?</system_note>', ' ', user_input, flags=re.DOTALL).strip()
    n_words = len(text.split())

    if DEEP_KEYWORDS_RE.search(text):
        return 'deep'
    if n_words <= FAST_MAX_WORDS and FAST_KEYWORDS_RE.search(text):
        return 'fast'
    # По умолчанию — fast, если запрос короткий (< 5 слов)
    if n_words < 5:
        return 'fast'
    return 'deep'  # для всего остального

def run_agent_loop(user_task: str, messages: list = None, max_steps: int = MAX_STEPS,
                   input_queue: queue.Queue = None,
                   memory_manager: MemoryManager = None,
                   soul_manager: SoulManager = None,
                   tts_manager: TTSManager = None,
                   emotion_bridge: EmotionBridge = None,
                   executor: ActionExecutor = None,
                   inner: Optional[str] = None) -> list:
    """
    Основной цикл выполнения одной задачи пользователя.
    Принимает все зависимости через параметры, чтобы быть тестируемым.
    Возвращает обновлённый список messages.

    :param inner: "autonomy"/"reflection" — внутренний ход: Юи думает про себя (текст не
        озвучивается, в истории помечен <inner_thought>), говорит вслух через speak_aloud,
        сама решает, когда закончить (task_complete). Прерывается, как только пользователь
        что-то сказал. Вызывающий код ставит executor.inner_mode на время хода.
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

    # 1. Системный промпт СТАТИЧЕН (см. комментарий над prompt.SYSTEM_PROMPT) —
    # это не зависит от soul_patch/памяти и не требует пересборки каждый ход.
    system_prompt = build_system_prompt()

    # 2. soul_patch (чтение нескольких маленьких файлов) и get_auto_context
    # (векторный поиск — encode на CPU, самая долгая часть этой пары) друг от
    # друга не зависят, поэтому считаем их параллельно, а не последовательно.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _prep_pool:
        soul_future = _prep_pool.submit(soul_manager.generate_soul_patch)
        auto_mem_future = _prep_pool.submit(memory_manager.get_auto_context, user_task)
        soul_patch = soul_future.result()
        auto_mem = auto_mem_future.result()

    # Всё изменчивое (время, железо, статус памяти, soul patch, найденный
    # контекст) едет в хвост — в user-сообщение, а не в системный промпт.
    user_input_final = inject_dynamic_context(user_task, auto_mem, soul_patch=soul_patch)

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
        # Скриншоты прошлых ходов уже описаны в ответах агента — выкидываем сами картинки
        strip_images(messages)
        messages[0]["content"] = system_prompt
        messages.append({"role": "user", "content": user_input_final})

    # 4. Своя воля: хочет ли она вообще за это браться (один раз на сообщение
    # пользователя, до всех итераций). Решение уходит в каждый запрос этого хода
    # эфемерной подсказкой, в историю не сохраняется.
    will_message = None
    if WILL_CHECK_ENABLED and not inner:  # внутренний ход — её собственная инициатива, спрашивать нечего
        agent_is_working.set()
        decision, reason = check_willingness(messages)
        agent_is_working.clear()
        print(f"[WILL] Решение: {decision}{' — ' + reason if reason else ''}")
        will_message = will_note(decision, reason)

    request_tools = inner_tools(AUTONOMY_ALLOW_PC_CONTROL) if inner else TOOLS
    continue_note = None  # внутренний ход: подсказка "продолжай думать или task_complete" (эфемерная)

    # 5. Основной цикл итераций
    try:
        for step in range(1, max_steps + 1):
            print(f"\n--- ИТЕРАЦИЯ {step}{f' ({inner})' if inner else ''} ---")

            # Сжатие контекста (если нужно)
            messages = compress_context(messages, memory_manager)

            # Внутренний ход уступает пользователю: заговорил — заканчиваем размышления,
            # его реплика останется в очереди и будет обработана обычным ходом.
            if inner and input_queue is not None and any(
                    item[0] in ("text", "voice") for item in list(input_queue.queue)):
                print(f"\n[INNER] Пользователь заговорил — Юи прерывает размышления ({inner}).")
                save_session(messages)
                return messages

            # Проверка очереди на новые сообщения от пользователя (интеррапты)
            if input_queue is not None and not inner:
                injected_texts = []
                while not input_queue.empty():
                    try:
                        source, new_input, metadata = input_queue.get_nowait()
                        if new_input == "EXIT" or source not in ("text", "voice"):
                            continue  # поставленные в очередь размышления после разговора уже не к месту
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
            self_reported_emotion = None
            success = False

            # Решаем один раз за шаг, а не на каждую попытку: иначе ретраи
            # после сетевой ошибки будут "перебрасывать монетку" заново.
            silence_nudge = not inner and random.random() < SILENCE_NUDGE_CHANCE
            if silence_nudge:
                print("[SYSTEM] (тихая подсказка: можно промолчать, если не хочется отвечать)")

            for attempt in range(MAX_RETRIES):
                agent_is_working.set()
                # Определяем режим
                mode = 'deep' if inner else classify_request(user_task)
                if mode == 'fast':
                    temperature = FAST_TEMPERATURE   # 0.1
                    max_tokens = FAST_MAX_TOKENS     # 256
                else:
                    temperature = DEEP_TEMPERATURE   # 0.6
                    max_tokens = DEEP_MAX_TOKENS     # 2048

                # Подсказка про stay_silent добавляется ТОЛЬКО в этот запрос,
                # в постоянную историю (messages) она не попадает.
                ephemeral = (([will_message] if will_message else [])
                             + ([SILENCE_NUDGE_MESSAGE] if silence_nudge else [])
                             + ([continue_note] if continue_note else []))
                request_messages = messages + ephemeral

                # В payload:
                payload = {
                    "messages": request_messages,
                    "tools": request_tools,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_k": DEFAULT_TOP_K,
                    "top_p": DEFAULT_TOP_P,
                    "repeat_penalty": DEFAULT_REPEAT_PENALTY,
                    "stop": STOP_TOKENS,
                    "stream": True
                }

                try:
                    response = SESSION.post(LLM_API_URL, json=payload, stream=True, timeout=LLM_TIMEOUT)
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

                    final_reply, reasoning, tool_calls, self_reported_emotion = parser.finalize()
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
                assistant_idx = len(messages)
                messages, task_complete, stayed_silent = executor.execute_tool_calls(tool_calls, messages, agent_is_working)
                # Текст, написанный вместе с вызовом инструмента, раньше терялся (content=""):
                # в обычном ходе это сказанное вслух, во внутреннем — сама мысль.
                if raw_reply.strip() and assistant_idx < len(messages):
                    thought = re.sub(r'</?inner_thought>', '', raw_reply).strip()
                    messages[assistant_idx]["content"] = (
                        f"<inner_thought>{thought}</inner_thought>" if inner else raw_reply)
                    if inner:
                        print(f"\n[YUI ДУМАЕТ ({inner})]: {speech_text(raw_reply)}")
                continue_note = None
                if task_complete or stayed_silent:
                    # Сохраняем факты и сессию. При stay_silent пользователь просто
                    # не получит ответа в этом ходу — это осознанный выбор агента,
                    # а не сбой.
                    extract_and_save_facts(messages[1:], memory_manager)
                    save_session(messages)
                    return messages
                # Иначе продолжаем цикл (следующая итерация)
                continue

            # Если tool_calls нет — финализируем ответ. Эмоцию (self-report из
            # <emotion>, либо эвристика по ключевым словам как фолбэк) executor
            # определяет и отправляет сам — дублировать здесь не нужно.
            should_exit = executor.finalize_response(
                raw_reply, reasoning, messages, agent_is_working,
                self_reported_emotion=self_reported_emotion
            )
            if should_exit:
                # Пустой ответ — выходим
                extract_and_save_facts(messages[1:], memory_manager)
                save_session(messages)
                return messages

            if inner:
                # Мысль без инструментов: помечаем её как мысль про себя (чтобы потом не путать
                # со сказанным пользователю). Первая такая мысль — ещё не конец: даём подумать
                # дальше. Вторая подряд без действий — размышление исчерпано (модель часто
                # пишет "можно завершать" текстом вместо task_complete).
                thought = re.sub(r'</?inner_thought>', '', raw_reply).strip()
                messages[-1]["content"] = f"<inner_thought>{thought}</inner_thought>"
                if continue_note is not None:
                    extract_and_save_facts(messages[1:], memory_manager)
                    save_session(messages)
                    return messages
                continue_note = INNER_CONTINUE_NOTE
                continue

            # Если есть финальный ответ без инструментов — сохраняем и завершаем цикл
            if raw_reply and not tool_calls:
                extract_and_save_facts(messages[1:], memory_manager)
                save_session(messages)
                return messages

        # Если цикл завершился по максимуму шагов
        print(f"[SYSTEM] Достигнут лимит шагов{f' ({inner})' if inner else ''}, завершаю.")
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

    # Автономия (если включена). agent_is_working/tts_active_event передаём, чтобы
    # фоновая мысль не отнимала слот у llama-server и не перебивала Юи на полуслове.
    # Готовая мысль возвращается через input_queue — историю меняет только основной поток.
    autonomy = None
    if ENABLE_AUTONOMY:
        autonomy = AutonomyManager(
            input_queue=input_queue,
            agent_is_working=agent_is_working,
            tts_active=tts_active_event
        )
        autonomy.start()

    # Фоновая рефлексия / консолидация памяти RAG 2.0 (если включена)
    reflection = None
    if ENABLE_REFLECTION:
        reflection = ReflectionManager(memory_manager=memory_mgr, input_queue=input_queue,
                                       agent_is_working=agent_is_working)
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

    # Прошлую сессию грузим сразу, а не на первой реплике: размышлениям Юи нужна
    # история уже до того, как пользователь что-то скажет.
    messages = load_session()
    if messages:
        print("[SYSTEM] Обнаружена предыдущая сессия. Восстановление...")
        messages[0]["content"] = build_system_prompt()  # в файле мог остаться старый промпт
    inner_managers = {"autonomy": autonomy, "reflection": reflection}
    inner_max_steps = {"autonomy": AUTONOMY_MAX_STEPS, "reflection": REFLECTION_MAX_STEPS}
    try:
        print("\n[YUI SYSTEM] Агент запущен. Пиши текст или говори в микрофон.")

        while True:
            source, user_input, metadata = input_queue.get()
            if user_input == "EXIT":
                break

            if source in inner_managers:
                # Внутренний ход: Юи размышляет сама (автономия/рефлексия) — в несколько шагов,
                # с инструментами; мысли видны в консоли, вслух — только speak_aloud.
                # Активностью пользователя это не считается.
                manager = inner_managers[source]
                if manager is None or not manager.is_due():
                    continue  # пока задача ждала в очереди, пользователь успел заговорить
                print(f"\n[YUI INNER: {source}] Юи размышляет сама...")
                executor.inner_mode = source
                try:
                    messages = run_agent_loop(
                        user_task=user_input,
                        messages=messages,
                        max_steps=inner_max_steps[source],
                        input_queue=input_queue,
                        memory_manager=memory_mgr,
                        soul_manager=soul_mgr,
                        tts_manager=tts_mgr,
                        emotion_bridge=emotion_br,
                        executor=executor,
                        inner=source
                    )
                finally:
                    executor.inner_mode = None
                print(f"[YUI INNER: {source}] Размышления закончены.")
                continue

            mark_activity()
            if autonomy:
                autonomy.on_user_activity()

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
            mark_activity()  # отсчёт тишины — с конца ответа Юи, а не с момента вопроса

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