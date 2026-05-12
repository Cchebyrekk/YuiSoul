import requests
import threading
import json
import re
import time
from pc_interaction.control import ComputerControl
from core.session_manager import load_session, save_session, clear_session
from core.prompt_builder import build_system_prompt
from core.autonomy_manager import AutonomyManager
from core.soul_manager import SoulManager

control = ComputerControl()
agent_is_working = threading.Event()

# --- CONFIGURATION ---
ENABLE_AUTONOMY = False

def parse_tool_calls(raw_text: str) -> list[dict]:
    """Парсит JSON (объекты и массивы), очищает от Markdown и нативного формата"""
    tools = []
    
    # 1. Очищаем от Markdown-оберток (```json ... ```)
    raw_text = re.sub(r'```json\s*', '', raw_text)
    raw_text = re.sub(r'```\s*', '', raw_text)
    
    # 2. Проверяем нативный формат Gemma
    gemma_match = re.search(r'<\|?tool_call\|?>call:\s*(\w+)\{(.+?)\}<\|?tool_call\|?>', raw_text, re.DOTALL)
    if gemma_match:
        tool_name = gemma_match.group(1)
        params_str = gemma_match.group(2).strip()
        params = {}
        pairs = re.findall(r'(\w+)\s*:\s*"?([^"}]+)"?', params_str)
        for key, val in pairs:
            params[key.strip()] = val.strip()
        if params:
            return [{"tool": tool_name, "params": params}]

    # 3. Пытаемся распарсить весь очищенный текст как стандартный JSON
    try:
        parsed = json.loads(raw_text.strip())
        if isinstance(parsed, list):
            # Если модель вернула массив [{"tool":...}, {"tool":...}]
            return [item for item in parsed if isinstance(item, dict) and "tool" in item]
        elif isinstance(parsed, dict) and "tool" in parsed:
            # Если вернула один объект {"tool":...}
            return [parsed]
    except json.JSONDecodeError:
        pass # Если не парсится целиком, падаем в глубокий поиск

    # 4. Глубокий поиск (выковыривает JSON из любого мусора)
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw_text):
        next_brace = raw_text.find('{', idx)
        if next_brace == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(raw_text[next_brace:])
            if "tool" in obj:
                tools.append(obj)
            idx = next_brace + end_idx
        except json.JSONDecodeError:
            idx = next_brace + 1
            
    return tools

def compress_context(messages: list) -> list:
    """Жесткая резка контекста с учетом буфера под 100k токенов"""
    MAX_CONTEXT_CHARS = 60000 
    
    if len(str(messages)) < MAX_CONTEXT_CHARS:
        return messages

    print("\n[SYSTEM WARNING] Контекст приближается к лимиту генерации. Резка истории.")
    
    system_msg = messages[0]
    # Сохраняем последние 8 сообщений (4 полных цикла инструмент-ответ)
    tail_msgs = messages[-8:]
    
    new_messages = [system_msg]
    new_messages.append({
        "role": "user", 
        "content": "<system_warning>Контекст переполнен. Старые данные извлечены в /memory. Продолжай с текущего состояния.</system_warning>"
    })
    new_messages.extend(tail_msgs)
    
    return new_messages

def format_raw_gemma_prompt(messages: list) -> str:
    """Конвертирует список сообщений в нативный формат Gemma 4 без прослойки OpenAI"""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            prompt += f"<start_of_turn>system\n{content}<end_of_turn>\n"
        elif role == "user":
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            
    # Добавляем триггер для генерации модели
    prompt += "<start_of_turn>model\n"
    return prompt

def extract_and_save_facts(history: list):
    """Извлекает факты с агрессивной фильтрацией мусора от 4B модели"""
    from memory.memory_manager import MemoryManager
    mm = MemoryManager()
    
    # Получаем дерево для передачи в промпт
    tree = mm._get_tree() 
    
    # --- AGGRESSIVE FILTERING OF SYSTEM EVENTS ---
    # Отрезаем результаты работы инструментов, чтобы 4B модель не пыталась пере-сохранить то, что уже сохранено
    cleaned_history = []
    for msg in history:
        clean_content = re.sub(r'<system_event>.*?</system_event>', '', msg.get("content", ""), flags=re.DOTALL)
        if clean_content.strip():
            cleaned_history.append({"role": msg["role"], "content": clean_content.strip()})
        
    extraction_prompt = mm.get_fact_extraction_prompt(cleaned_history, tree)
    
    try:
        raw_prompt = format_raw_gemma_prompt([{"role": "user", "content": extraction_prompt}])
        response = requests.post("http://127.0.0.1:8080/completion", json={
            "prompt": raw_prompt,
            "n_predict": 2048,
            "temperature": 0.2,
            "stop": ["<start_of_turn>", "<end_of_turn>"]
        }, timeout=60.0)
        
        raw_facts = response.json().get("content", "").strip()
        
        # Убираем остатки XML
        raw_facts = re.sub(r'<[^>]+>', '', raw_facts).strip()
        
        if not raw_facts or raw_facts.upper() == "NULL":
            return

        # --- AGGRESSIVE FILTERING ---
        # Список слов-маркеров рассуждений модели (на русском и английском)
        garbage_keywords = [
            'анализ:', 'источник:', 'шаг:', 'формат:', 'итоговый', 'вывод:', 
            'факты:', 'самопроверка:', 'финальный', 'правила:', 'пример:', 
            'analysis:', 'step', 'conclusion:', 'note:', 'format:'
        ]
        
        valid_facts = set() # Используем SET для мгновенного убийства дублей внутри одной генерации
        
        for line in raw_facts.split('\n'):
            line = line.strip("- *").strip()
            if not line: continue
            
            # Отсекаем строки, которые начинаются со слов-маркеров (без учета регистра)
            if any(line.lower().startswith(kw) for kw in garbage_keywords):
                continue
                
            # Парсим ТОЛЬКО валидный формат [путь] текст
            match = re.match(r'^\[(.*?)\]\s*(.*)', line)
            if match:
                path = match.group(1).strip()
                fact = match.group(2).strip()
                
                # Дополнительная проверка: факт не должен быть слишком коротким (случайные слова)
                if len(fact) > 5:
                    valid_facts.add( (path, fact) )
            # Фолбэк на misc/unsorted УДАЛЕН. Если модель не able выдать правильный формат - факт просто отбрасывается.
            
        # --- SAVE LOOP ---
        saved_count = 0
        for path, fact in valid_facts:
            mm.save_fact(path, fact)
            saved_count += 1
            
        if saved_count > 0:
            print(f"[SYSTEM] Факты ({saved_count} шт.) распределены по /memory.")
            
    except Exception as e:
        print(f"[SYSTEM ERROR] Извлечение фактов провалено: {e}")

def run_agent_loop(user_task: str, messages: list = None, max_steps: int = 10):
    print(f"[SYSTEM] Запуск агента. Задача: {user_task}")
    # --- ДИНАМИЧЕСКАЯ ДУША ---
    sm = SoulManager()
    current_soul_patch = sm.generate_soul_patch()
    current_system_prompt = build_system_prompt(soul_patch=current_soul_patch)
    # --------------------------
    
    if messages is None:
        existing_session = load_session()
        if existing_session:
            print("[SYSTEM] Обнаружена предыдущая сессия. Восстановление...")
            messages = existing_session
            messages[0]["content"] = current_system_prompt
            messages.append({"role": "user", "content": user_task})
        else:
            messages = [{"role": "system", "content": current_system_prompt}, {"role": "user", "content": user_task}]
    else:
        messages[0]["content"] = current_system_prompt
        messages.append({"role": "user", "content": user_task})
    
    try:
        for step in range(1, max_steps + 1):
            print(f"\n--- ИТЕРАЦИЯ {step} ---")
            
            # Контроль переполнения перед каждым шагом
            messages = compress_context(messages)
            # --- RAW COMPLETION (Bypass OpenAI Template) ---
            raw_reply = ""
            max_retries = 3
            
            # Превращаем историю в одну сырую строку
            raw_prompt = format_raw_gemma_prompt(messages)
            
            for attempt in range(max_retries):
                agent_is_working.set() 
                response = requests.post("http://127.0.0.1:8080/completion", json={
                    "prompt": raw_prompt,
                    "n_predict": 8192,
                    "temperature": 0.4,
                    "stop": ["<start_of_turn>", "<end_of_turn>", "</response_format>", "<|eot_id|>"] 
                })
                agent_is_working.clear()

                # --- ЗАЩИТА ОТ ПЕРЕПОЛНЕНИЯ ЖЕЛЕЗА ---
                if response.status_code == 400 and "exceed_context_size" in response.text:
                    print("\n[SYSTEM CRITICAL] Физический лимит токенов превышен (llama.cpp 400).")
                    print("[SYSTEM] Инициация экстренной резки контекста...")
                    
                    # Сохраняем системный промпт и последние 2 сообщения (чтобы модель помнила, что делала)
                    system_msg = messages[0]
                    tail_msgs = messages[-2:]
                    middle_msgs = messages[1:-2]
                    
                    # Выжимаем факты из отрезанной части
                    if middle_msgs:
                        extract_and_save_facts(middle_msgs)
                    
                    # Пересобираем контекст
                    messages = [system_msg]
                    messages.append({
                        "role": "user", 
                        "content": "<system_warning>КРИТИЧЕСКОЕ ПЕРЕПОЛНЕНИЕ. Старая история экстренно извлечена в память. Забудь старый контекст и продолжай исходя из текущего состояния.</system_warning>"
                    })
                    messages.extend(tail_msgs)
                    
                    print("[SYSTEM] Контекст обрезан. Перезапуск итерации...")
                    time.sleep(1)
                    continue # Возвращаемся в начало цикла for с обрезанным контекстом
                # -----------------------------------------

                if response.status_code != 200:
                    print(f"[ERROR] Сервер упал: {response.text}")
                    break
                    
                # У нативного эндпоинта другой путь к контенту
                raw_reply = response.json().get("content", "").strip()
                
                # --- TAG NORMALIZER (Решение конфликта нативных весов Gemma) ---
                # Переводим любые нативные токены размышлений в наш стандартный формат
                raw_reply = raw_reply.replace("<|thought|>", "<thought>").replace("<|/thought|>", "</thought>")
                raw_reply = raw_reply.replace("<|thinking|>", "<thought>").replace("<|/thinking|>", "</thought>")
                raw_reply = re.sub(r'<\|[^>]*\|?>', '', raw_reply)
                
                # Очистка утечек нативного шаблона Gemma
                raw_reply = raw_reply.replace("</response_format>", "").replace("<end_of_turn>", "")
                
                # Если модель взяла рекурсию тегов (спам <output><output>), схлопываем в один
                raw_reply = re.sub(r'(<output>\s*)+', '<output>', raw_reply)
                # -------------------------------------------------------------------

                if raw_reply:
                    break
                else:
                    print(f"[SYSTEM] Пустой ответ. Retry {attempt + 1}/{max_retries}...")
                    time.sleep(0.5)
            
            if not raw_reply:
                print("[SYSTEM] Модель упорно молчит. Сброс итерации.")
                continue
            
            # --- PRE-PROCESSOR (Санитария 4B модели) ---
            # 1. Убиваем рекурсию закрытых/открытых тегов (схлопываем любые повторы в один)
            raw_reply = re.sub(r'(</thought>\s*)+', '</thought>', raw_reply)
            raw_reply = re.sub(r'(<thought>\s*)+', '<thought>', raw_reply)
            raw_reply = re.sub(r'(</instrument_call>\s*)+', '</instrument_call>', raw_reply)
            raw_reply = re.sub(r'(<instrument_call>\s*)+', '<instrument_call>', raw_reply)
            raw_reply = re.sub(r'(</output>\s*)+', '</output>', raw_reply)
            raw_reply = re.sub(r'(<output>\s*)+', '<output>', raw_reply)

            # 2. Неявное закрытие тегов (если модель забыла закрыть)
            raw_reply = re.sub(r'(<thought>)(.*?)(?=(?:<instrument_call>|<output>))', r'\1\2</thought>', raw_reply, flags=re.DOTALL)
            raw_reply = re.sub(r'(<instrument_call>)(.*?)(?=(?:<output>|$))', r'\1\2</instrument_call>', raw_reply, flags=re.DOTALL)

            # 3. Удаление физических переносов строк внутри JSON
            raw_reply = re.sub(
                r'<instrument_call>(.*?)</instrument_call>', 
                lambda m: m.group(0).replace('\n', ' '), 
                raw_reply, 
                flags=re.DOTALL
            )
            # ---------------------------------------------

            print(f"[LLM RAW]: {raw_reply}")
            # -------------------------------------------------
            
            tool_list = parse_tool_calls(raw_reply)
            
            if tool_list:
                os_results = []
                for tool_data in tool_list:
                    agent_is_working.set()
                    tool_name = tool_data["tool"]
                    params = tool_data.get("params", {})
                    os_result = "" # Инициализация во избежание UnboundLocalError
                    
                    if tool_name == "task_complete":
                        # Ищем текст внутри <output>. Если тег не закрыт, берем до конца строки.
                        output_match = re.search(r'<output>(.*?)(?:</output>|$)', raw_reply, re.DOTALL)
                        final_text = output_match.group(1).strip() if output_match else ""
                        
                        if final_text:
                            print(f"\n[YUI FINAL]: {final_text}")
                            
                        print(f"\n[SYSTEM] Агент завершил работу.")
                        print(f"[REASON]: {params.get('reason', 'Не указана')}")
                        
                        extract_and_save_facts(messages[1:])
                        
                        save_session(messages)
                        agent_is_working.clear()
                        return messages

                    # --- ИНСТРУМЕНТЫ ПАМЯТИ ---
                    elif tool_name == "save_memory":
                        from memory.memory_manager import MemoryManager
                        mm = MemoryManager()
                        os_result = mm.save_fact(params.get("path", "misc/default"), params.get("content", ""))
                        
                        # Проверка на конфликт (Серая зона)
                        if isinstance(os_result, dict) and os_result.get("status") == "conflict":
                            c = os_result
                            os_result = (
                                f"[MEMORY CONFLICT] В файле {c['path']} найден спорный факт (Схожесть: {c['similarity']}).\n"
                                f"Старый: {c['old_fact']}\n"
                                f"Новый: {c['new_fact']}\n"
                                f"Ты должна решить: это один и тот же факт (вызови save_memory с тем же путем, чтобы ПЕРЕЗАПИСАТЬ старый на новый) "
                                f"или это разные вещи (вызови task_complete, система оставит старый факт и допишет новый отдельно)."
                            )
                        
                    elif tool_name == "search_memory":
                        from memory.memory_manager import MemoryManager
                        mm = MemoryManager()
                        os_result = mm.search_facts(params.get("query", ""))
                    # ----------------------------

                    else:
                        print(f"[ACTION] -> {tool_name} | {params}")
                        os_result = control.execute_action(tool_name, params)
                        print(f"[OS LOG] {os_result}")
                        
                        if tool_name == "open_app":
                            print("[SYSTEM] Ждем фокуса окна...")
                            time.sleep(3)
                        else:
                            time.sleep(0.8)
                    
                    # Обязательно собираем результат КАЖДОЙ итерации
                    os_results.append(os_result)
                
                # Обновление контекста происходит после выполнения всей цепочки инструментов
                messages.append({"role": "assistant", "content": raw_reply})
                # Жесткая обертка системного ответа, чтобы модель не путала его с речью человека
                messages.append({"role": "user", "content": f"<system_event>Результат выполнения: {os_results}</system_event>"})
                agent_is_working.clear()
                
            else:
                # Защита от пустого EOS токена
                if not raw_reply:
                    print("[SYSTEM] Агент выдал пустой ответ.")
                    return messages
                
                # Если инструментов нет — чистим мусор от тегов перед выводом
                clean_output = re.sub(r'<thought>.*?</thought>', '', raw_reply, flags=re.DOTALL).strip()
                clean_output = re.sub(r'<instrument_call>.*?</instrument_call>', '', clean_output, flags=re.DOTALL).strip()
                clean_output = clean_output.replace("<output>", "").replace("</output>", "").strip()

                if clean_output:
                    print(f"\n[YUI FINAL]: {clean_output}")
                else:
                    print("[SYSTEM] Агент выдал невалидный ответ (только мысли без output).")
                
                extract_and_save_facts(messages[1:])
                return messages
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Ручная остановка (Ctrl+C). Спасаю факты и контекст...")
        if len(messages) > 1:
            extract_and_save_facts(messages[1:])
        save_session(messages)
        print("[SYSTEM] Сессия сохранена.")
        return messages
    except Exception as e:
        print(f"\n[SYSTEM] Фатальная ошибка: {e}. Спасаю факты и контекст...")
        if len(messages) > 1:
            extract_and_save_facts(messages[1:])
        save_session(messages)
        return messages

if __name__ == "__main__":
    current_messages = None 
    session_state = {"messages": []}
    
    autonomy = None
    
    if ENABLE_AUTONOMY:
        autonomy = AutonomyManager(work_flag=agent_is_working, session_state=session_state, check_interval=600, idle_threshold=300)
        autonomy.start()
        
    try:
        while True:
            try:
                user_input = input()
            except KeyboardInterrupt:
                print("\n[SYSTEM] Завершение работы YUI.")
                break
            
            if not user_input.strip(): continue
            current_messages = run_agent_loop(user_input, current_messages)
            session_state["messages"] = current_messages
            
    except Exception as e:
        print(f"\n[SYSTEM FATAL] Падение основного цикла: {e}")
    finally:
        if autonomy is not None:
            autonomy.stop() 
        if current_messages:
            print("[SYSTEM] Финальное сохранение сессии...")
            save_session(current_messages)