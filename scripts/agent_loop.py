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
from core.tts_manager import TTSManager
from memory.memory_manager import MemoryManager
from core.tool_registry import TOOLS, build_registry
from core.prompt_builder import get_dynamic_state # Добавь этот импорт в начало файла

mm = MemoryManager() 
tts_engine = TTSManager(rvc_model_name="yui_girl.pth", rvc_api_url="http://127.0.0.1:7865") 
TOOL_REGISTRY = build_registry(mm) # Инициализация реестра

control = ComputerControl()
agent_is_working = threading.Event()

ENABLE_AUTONOMY = False
MAX_CONTEXT_CHARS = 60000

def compress_context(messages: list) -> list:
    TAIL_CHARS_LIMIT = MAX_CONTEXT_CHARS/4 
    total_chars = len(str(messages))
    if total_chars < MAX_CONTEXT_CHARS:
        return messages

    print(f"\n[SYSTEM WARNING] Контекст достиг {total_chars} символов. Резка истории.")
    system_msg = messages[0]
    rest_msgs = messages[1:]
    
    tail_msgs = []
    tail_chars = 0
    for msg in reversed(rest_msgs):
        msg_chars = len(str(msg))
        if tail_chars + msg_chars > TAIL_CHARS_LIMIT:
            break
        tail_msgs.insert(0, msg)
        tail_chars += msg_chars
    
    deleted_msgs = rest_msgs[:len(rest_msgs) - len(tail_msgs)]
    if deleted_msgs:
        extract_and_save_facts(deleted_msgs)
    
    new_messages = [system_msg]
    new_messages.append({
        "role": "user", 
        "content": "<system_warning>Контекст переполнен. Продолжай с текущего состояния.</system_warning>"
    })
    new_messages.extend(tail_msgs)
    save_session(new_messages)
    return new_messages

def extract_and_save_facts(history: list):
    tree = mm._get_tree() 
    cleaned_history = []
    for msg in history:
        clean_content = msg.get("content", "")
        clean_content = re.sub(r'<system_event>.*?</system_event>', '', clean_content, flags=re.DOTALL)
        if clean_content.strip():
            cleaned_history.append({"role": msg["role"], "content": clean_content.strip()})
        
    extraction_prompt = mm.get_fact_extraction_prompt(cleaned_history, tree)
    
    try:
        response = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={
            "messages": [{"role": "user", "content": extraction_prompt}],
            "max_tokens": 2048,
            "temperature": 0.2,
        }, timeout=180.0)
        
        raw_facts = response.json()["choices"][0]["message"]["content"].strip()
        raw_facts = re.sub(r'<[^>]+>', '', raw_facts).strip()
        
        if not raw_facts or raw_facts.upper() == "NULL":
            return

        garbage_keywords = ['анализ:', 'источник:', 'шаг:', 'формат:', 'итоговый', 'вывод:', 'факты:', 'самопроверка:']
        valid_facts = set()
        
        for line in raw_facts.split('\n'):
            line = line.strip("- *").strip()
            if not line: continue
            if any(line.lower().startswith(kw) for kw in garbage_keywords): continue
            match = re.match(r'^\[(.*?)\]\s*(.*)', line)
            if match:
                path, fact = match.group(1).strip(), match.group(2).strip()
                if len(fact) > 5: valid_facts.add((path, fact))
            
        saved_count = 0
        for path, fact in valid_facts:
            mm.save_fact(path, fact)
            saved_count += 1
        if saved_count > 0: print(f"[SYSTEM] Факты ({saved_count} шт.) распределены.")
            
    except Exception as e:
        print(f"[SYSTEM ERROR] Извлечение фактов провалено: {e}")

def run_agent_loop(user_task: str, messages: list = None, max_steps: int = 10):
    print(f"[SYSTEM] Запуск агента. Задача: {user_task}")
    
    sm = SoulManager()
    current_soul_patch = sm.generate_soul_patch()
    current_system_prompt = build_system_prompt(soul_patch=current_soul_patch, context_memory="") 
    
    auto_mem = mm.get_auto_context(user_task)
    dynamic_state = get_dynamic_state() # Получаем время и температуру
    user_input_final = user_task
    
    if auto_mem:
        print(f"\n[SYSTEM AUTO-MEMORY EXTRACTED]:\n{auto_mem}\n")
        user_input_final = (
            f"{user_task}\n\n"
            f"<injected_context>\n{dynamic_state}\n\n"
            f"ВНИМАНИЕ! Система УЖЕ нашла в памяти ответ. "
            f"ЗАПРЕЩЕНО вызывать search_memory. Используй ТОЛЬКО эти данные:\n{auto_mem}\n</injected_context>"
        )
    else:
        # Если памяти нет, всё равно инжектим время и железо, чтобы модель их видела
        user_input_final = (
            f"{user_task}\n\n"
            f"<injected_context>\n{dynamic_state}\n</injected_context>"
        )

    if messages is None:
        existing_session = load_session()
        if existing_session:
            print("[SYSTEM] Обнаружена предыдущая сессия. Восстановление...")
            messages = existing_session
            messages[0]["content"] = current_system_prompt
            messages.append({"role": "user", "content": user_input_final})
        else:
            messages = [{"role": "system", "content": current_system_prompt}, {"role": "user", "content": user_input_final}]
    else:
        messages[0]["content"] = current_system_prompt
        messages.append({"role": "user", "content": user_input_final})
    
    try:
        for step in range(1, max_steps + 1):
            print(f"\n--- ИТЕРАЦИЯ {step} ---")
            messages = compress_context(messages)
            
            max_retries = 3
            raw_reply = ""
            reasoning_reply = ""
            tool_calls = []
            
            for attempt in range(max_retries):
                agent_is_working.set() 
                
                payload = {
                    "messages": messages,
                    "tools": TOOLS, # <--- ПЕРЕДАЕМ СХЕМУ В API
                    "max_tokens": 8192,
                    "temperature": 0.4,
                    "stream": True
                }
                
                response = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload, stream=True, timeout=120.0)

                if response.status_code == 400:
                    if "exceed_context_size" in response.text:
                        print("\n[SYSTEM CRITICAL] Лимит токенов превышен. Резка...")
                        system_msg = messages[0]
                        tail_msgs = messages[-2:]
                        middle_msgs = messages[1:-2]
                        if middle_msgs: extract_and_save_facts(middle_msgs)
                        messages = [system_msg] + [{"role": "user", "content": "<system_warning>КРИТИЧЕСКОЕ ПЕРЕПОЛНЕНИЕ.</system_warning>"}] + tail_msgs
                        agent_is_working.clear()
                        time.sleep(1)
                        continue 

                if response.status_code != 200:
                    agent_is_working.clear()
                    print(f"[ERROR] Сервер упал: {response.text}")
                    break
                    
                raw_reply = ""
                tool_calls = []
                
                # Ассемблирование стриминга (с поддержкой потоковых tool_calls)
                current_tc = None
                print("[LLM STREAM]: ", end="", flush=True)
                
                for line in response.iter_lines():
                    if not line: continue
                    decoded_line = line.decode('utf-8')
                    if not decoded_line.startswith('data: '): continue
                    json_str = decoded_line[6:]
                    if json_str.strip() == '[DONE]': break
                    
                    try:
                        chunk_data = json.loads(json_str)
                        delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                        
                        # Перехват нативных скрытых размышлений (если llama.cpp их отделил)
                        if 'reasoning_content' in delta and delta['reasoning_content']:
                            token = delta['reasoning_content']
                            reasoning_reply += token
                            # Выводим в консоль серым цветом, чтобы не путать с ответом
                            print(f"\033[90m{token}\033[0m", end="", flush=True)
                            
                        # Обработка текста
                        if 'content' in delta and delta['content']:
                            token = delta['content']
                            raw_reply += token
                            print(token, end="", flush=True)
                            
                        # Обработка потоковых tool_calls (OpenAI format)
                        if 'tool_calls' in delta:
                            for tc_chunk in delta['tool_calls']:
                                idx = tc_chunk.get("index", 0)
                                # Расширяем список, если пришел новый индекс
                                while len(tool_calls) <= idx:
                                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                
                                tc = tool_calls[idx]
                                if tc_chunk.get("id"): tc["id"] = tc_chunk["id"]
                                if tc_chunk.get("function", {}).get("name"): tc["function"]["name"] += tc_chunk["function"]["name"]
                                if tc_chunk.get("function", {}).get("arguments"): tc["function"]["arguments"] += tc_chunk["function"]["arguments"]
                                
                    except json.JSONDecodeError:
                        continue
                
                print() # Перенос после стрима
                agent_is_working.clear()
                
                # ======================================================
                # ПРЕПРОЦЕССИНГ: РАЗДЕЛЕНИЕ ПО ТЕГУ ВЫХОДА ИЗ МЫСЛЕЙ
                # ======================================================
                
                # 1. Режем по закрывающему тегу </thought>
                if '</thought>' in raw_reply:
                    parts = raw_reply.split('</thought>', 1)
                    # Всё, что до тега — это мысли (даже если модель забыла открыть <thought>)
                    reasoning_reply += parts[0].strip()
                    # Всё, что после — финальный ответ
                    raw_reply = parts[1].strip()
                else:
                    # Фолбэк: если модель вообще не вывела </thought>, но вывела <output>
                    # значит весь текст до <output> был размышлением
                    if '<output>' in raw_reply and not reasoning_reply:
                        out_match = re.search(r'(<output>)', raw_reply)
                        if out_match:
                            reasoning_reply += raw_reply[:out_match.start()].strip()
                            # raw_reply оставляем как есть, экстрактор <output> ниже его заберет

                # 2. АГРЕССИВНАЯ ВЫРЕЗКА QWEN-ПРЕАМБУЛ (извлекли в мысли, но из ответа вырезать)
                # Модель может спамить преамбулы до выхода из размышлений, это уже ушло в reasoning_reply.
                # Но если она написала преамбулу после </thought>, выжигаем:
                preamble_patterns = [
                    r'^\s*Here\'s a thinking process:.*?(?=\n|<output>|$)',
                    r'^\s*Thinking Process:.*?(?=\n|<output>|$)',
                    r'^\s*The user wants me to perform.*?(?=\n|<output>|$)'
                ]
                for pattern in preamble_patterns:
                    raw_reply = re.sub(pattern, '', raw_reply, flags=re.DOTALL).strip()

                # 3. Вырезаем оставшиеся огрызки тегов <output>, если модель забыла их закрыть
                raw_reply = re.sub(r'</?output>', '', raw_reply).strip()
                
                # Вывод очищенного результата
                if reasoning_reply:
                    print(f"\n[LLM REASONING EXTRACTED]: {reasoning_reply.strip()}")
                print(f"[LLM RAW CLEANED]: {raw_reply}")

                # ======================================================
                # ОМНИ-ПАРСЕР (Фолбэк, если llama.cpp не перехватил вызов)
                # ======================================================
                if not tool_calls:
                    # Формат 1: <tool_code>tool_name(param="val")</tool_code> (Выдала Qwen сейчас)
                    tc_match = re.search(r'<tool_code>\s*(\w+)\((.*?)\)\s*</tool_code>', raw_reply, re.DOTALL)
                    if tc_match:
                        name = tc_match.group(1)
                        args_str = tc_match.group(2)
                        params = {}
                        # Парсим key="value" или key='value'
                        for k, v in re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
                            params[k] = v
                        tool_calls.append({
                            "id": "fallback_0", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(params)}
                        })
                    
                    # Формат 2: ✿function_call✿: {...}
                    elif "✿function_call✿" in raw_reply:
                        match = re.search(r'✿function_call✿:\s*({.*?})', raw_reply, re.DOTALL)
                        if match:
                            try:
                                parsed = json.loads(match.group(1))
                                tool_calls.append({
                                    "id": "fallback_1", "type": "function",
                                    "function": {"name": parsed.get("name"), "arguments": json.dumps(parsed.get("arguments", {}))}
                                })
                            except: pass
                            
                    # Формат 3: Глубокий поиск JSON {"tool": "...", "params": {...}}
                    if not tool_calls:
                        decoder = json.JSONDecoder()
                        idx = 0
                        while idx < len(raw_reply):
                            next_brace = raw_reply.find('{', idx)
                            if next_brace == -1: break
                            try:
                                obj, end_idx = decoder.raw_decode(raw_reply[next_brace:])
                                if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
                                    t_name = obj.get("tool", obj.get("name"))
                                    t_params = obj.get("params", obj.get("arguments", {}))
                                    if isinstance(t_params, str):
                                        try: t_params = json.loads(t_params)
                                        except: t_params = {}
                                    tool_calls.append({
                                        "id": f"fallback_json_{len(tool_calls)}", "type": "function",
                                        "function": {"name": t_name, "arguments": json.dumps(t_params) if isinstance(t_params, dict) else "{}"}
                                    })
                                idx = next_brace + end_idx
                            except json.JSONDecodeError:
                                idx = next_brace + 1
                
                if raw_reply or tool_calls:
                    break
                else:
                    print(f"[SYSTEM] Пустой ответ. Retry {attempt + 1}/{max_retries}...")
                    time.sleep(0.5)
            
            if not raw_reply and not tool_calls:
                print("[SYSTEM] Модель упорно молчит. Сброс итерации.")
                continue
            
            print(f"[LLM RAW]: {raw_reply}")
            if tool_calls: print(f"[TOOL CALLS]: {json.dumps(tool_calls, indent=2)}")
            
            # --- ИСПОЛНЕНИЕ ИНСТРУМЕНТОВ ---
            if tool_calls:
                os_results = []
                assistant_msg = {"role": "assistant", "content": raw_reply, "tool_calls": tool_calls}
                messages.append(assistant_msg)
                
                for tc in tool_calls:
                    agent_is_working.set()
                    func_name = tc["function"]["name"]
                    func_args_str = tc["function"]["arguments"]
                    
                    try:
                        func_args = json.loads(func_args_str)
                    except json.JSONDecodeError:
                        func_args = {}
                    
                    # Спец-обработка task_complete
                    if func_name == "task_complete":
                        output_match = re.search(r'<output>(.*?)(?:</output>|$)', raw_reply, re.DOTALL)
                        final_text = output_match.group(1).strip() if output_match else ""
                        if final_text:
                            print(f"\n[YUI FINAL]: {final_text}")
                            tts_engine.speak(final_text)
                        print(f"\n[SYSTEM] Агент завершил работу. Причина: {func_args.get('reason', 'Не указана')}")
                        extract_and_save_facts(messages[1:])
                        save_session(messages)
                        agent_is_working.clear()
                        return messages

                    # ДИСПЕТЧЕР РЕЕСТРА
                    if func_name in TOOL_REGISTRY:
                        print(f"[ACTION] -> {func_name} | {func_args}")
                        os_result = TOOL_REGISTRY[func_name](**func_args)
                        print(f"[OS LOG] {os_result}")
                        
                        # Перехват флага task_complete (если модель передала его без инструмента)
                        if isinstance(os_result, str) and os_result.startswith("TASK_COMPLETE:"):
                            print(f"\n[SYSTEM] Агент завершил работу.")
                            extract_and_save_facts(messages[1:])
                            save_session(messages)
                            agent_is_working.clear()
                            return messages
                            
                        if func_name == "open_app":
                            print("[SYSTEM] Ждем фокуса окна...")
                            time.sleep(3)
                        else:
                            time.sleep(0.8)
                    else:
                        os_result = f"[ERROR] Неизвестный инструмент: {func_name}"
                        print(os_result)
                    
                    os_results.append(os_result)
                    
                    # Добавляем результат инструмента в формат OpenAI
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(os_result)
                    })
                
                agent_is_working.clear()
                
            else:
                # Логика финального ответа без инструментов
                
                # 1. ОБЯЗАТЕЛЬНО сохраняем ответ ассистента в историю, 
                # иначе на следующей итерации он забудет, что уже ответил, и зациклится.
                messages.append({"role": "assistant", "content": raw_reply})
                
                if not raw_reply and not reasoning_reply:
                    print("[SYSTEM] Агент выдал пустой ответ.")
                    return messages
                
                # 2. Экстракция текста для TTS
                output_match = re.search(r'<output>(.*?)(?:</output>|$)', raw_reply, re.DOTALL)
                if output_match:
                    clean_output = output_match.group(1).strip()
                else:
                    clean_output = raw_reply.strip()
                
                # АГРЕССИВНАЯ ЗАЩИТА TTS: Вырезаем ЛЮБЫЕ XML-подобные теги 
                clean_output = re.sub(r'<[^>]+>', '', clean_output).strip()
                clean_output = clean_output.replace('```', '').strip()

                if clean_output:
                    print(f"\n[YUI FINAL]: {clean_output}")
                    tts_engine.speak(clean_output)
                else:
                    print("[SYSTEM] Агент выдал невалидный ответ (только мысли/код без текста).")
                
                # 3. Сохраняем факты и ПРИНУДИТЕЛЬНО выходим из цикла итераций
                extract_and_save_facts(messages[1:])
                save_session(messages)
                return messages
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Ручная остановка (Ctrl+C). Спасаю факты...")
        if len(messages) > 1: extract_and_save_facts(messages[1:])
        save_session(messages)
        return messages
    except Exception as e:
        print(f"\n[SYSTEM] Фатальная ошибка: {e}. Спасаю факты...")
        if len(messages) > 1: extract_and_save_facts(messages[1:])
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
            try: user_input = input()
            except KeyboardInterrupt:
                print("\n[SYSTEM] Завершение работы YUI.")
                break
            if not user_input.strip(): continue
            current_messages = run_agent_loop(user_input, current_messages)
            session_state["messages"] = current_messages if current_messages else []
    except Exception as e:
        print(f"\n[SYSTEM FATAL] Падение основного цикла: {e}")
    finally:
        if autonomy is not None: autonomy.stop() 
        if current_messages: save_session(current_messages)