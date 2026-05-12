import threading
import time
import datetime
import ctypes
import requests
import re
from core.soul_manager import SoulManager

class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ('cbSize', ctypes.c_uint),
        ('dwTime', ctypes.c_uint)
    ]

def get_idle_seconds() -> int:
    """Возвращает время бездействия пользователя в секундах (WinAPI)"""
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
    millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
    return millis // 1000

class AutonomyManager:
    def __init__(self, work_flag: threading.Event, session_state: dict, check_interval: int = 600, idle_threshold: int = 300):
        """
        :param session_state: Словарь вида {"messages": list}, ссылка на который обновляется в agent_loop.
        """
        self.work_flag = work_flag
        self.session_state = session_state
        self.check_interval = check_interval
        self.idle_threshold = idle_threshold
        self._running = False
        self._thread = None
        self.soul_manager = SoulManager()

    def _get_recent_context(self, max_chars: int = 400) -> str:
        """Извлекает последние действия из RAM-буфера сессии, отсекая системный промпт"""
        messages = self.session_state.get("messages")
        if not messages: return ""
        
        recent = messages[-6:] # Берем последние 3 цикла (запрос-ответ)
        context_lines = []
        total_len = 0
        
        for msg in reversed(recent):
            role = msg.get("role", "").upper()
            if role == "SYSTEM": continue # Отсекаем гигантский системный промпт
            
            content = msg.get("content", "")
            # Агрессивно чистим от XML, чтобы не тратить токены на логи инструментов
            content = re.sub(r'<[^>]+>', '', content).strip()
            if not content: continue
            
            if total_len + len(content) > max_chars: break
            context_lines.insert(0, f"{role}: {content}")
            total_len += len(content)
            
        return "\n".join(context_lines)

    def _build_autonomy_prompt(self, time_str: str, idle_min: int, soul_patch: str) -> str:
        recent_ctx = self._get_recent_context()
        ctx_block = f"<recent_context>\n{recent_ctx}\n</recent_context>\n" if recent_ctx else ""
        
        prompt = (
            f"<start_of_turn>user\n"
            f"Время: {time_str}. Ты в одиночестве уже {idle_min} мин.\n"
            f"{ctx_block}"
            f"{soul_patch}\n"
            f"Напиши короткую реплику в <thought>. Если хочешь что-то сказать вслух — в <output>. "
            f"ЗАПРЕЩЕНО: нумерованные списки, пункты, слово 'Анализ'. Только живая речь.\n"
            f"Если нечего говорить — <output>NULL<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        return prompt

    def _autonomy_loop(self):
        print("[AUTONOMY] Фоновый процесс инициализирован. Первая проверка через 60 сек...", flush=True)
        time.sleep(60) 
        
        while self._running:
            if not self._running: break

            try:
                # КРИТИЧЕСКАЯ ЗАЩИТА
                if self.work_flag.is_set():
                    time.sleep(30) 
                    continue

                # Безопасный вызов WinAPI
                try:
                    idle_sec = get_idle_seconds()
                except Exception as e:
                    print(f"[AUTONOMY] Ошибка WinAPI: {e}", flush=True)
                    idle_sec = 0 # Если WinAPI упал, считаем юзера активным
                
                print(f"[AUTONOMY DEBUG] Простой: {idle_sec}с / Порог: {self.idle_threshold}с", flush=True)

                if idle_sec < self.idle_threshold:
                    time.sleep(self.check_interval)
                    continue

                idle_min = idle_sec // 60
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"[AUTONOMY] Триггер! Пользователь молчит {idle_min} мин. Запрос спонтанной мысли...", flush=True)
                
                soul_patch = self.soul_manager.generate_soul_patch()
                if not soul_patch:
                    soul_patch = "<user_profile>Нет данных о пользователе.</user_profile>"

                raw_prompt = self._build_autonomy_prompt(time_str, idle_min, soul_patch)

                response = requests.post("http://127.0.0.1:8080/completion", json={
                    "prompt": raw_prompt,
                    "n_predict": 1048, 
                    "temperature": 0.8,
                    "stop": ["<start_of_turn>", "<end_of_turn>", "NULL"]
                }, timeout=30.0)

                if response.status_code != 200:
                    print(f"[AUTONOMY] Ошибка бэкенда: {response.status_code}", flush=True)
                    time.sleep(self.check_interval)
                    continue

                raw_reply = response.json().get("content", "").strip()
                
                thought_match = re.search(r'<thought>(.*?)</thought>', raw_reply, flags=re.DOTALL)
                if thought_match:
                    silent_thought = thought_match.group(1).strip()
                    if silent_thought:
                        print(f"[AUTONOMY] <Внутренняя мысль> {silent_thought}", flush=True)

                clean_output = re.sub(r'<thought>.*?</thought>', '', raw_reply, flags=re.DOTALL).strip()
                clean_output = clean_output.replace("<output>", "").replace("</output>", "").strip()

                if clean_output and clean_output != "NULL":
                    print(f"\n[YUI AUTONOMOUS] ({idle_min} мин. тишины) {clean_output}\n", flush=True)
                else:
                    print("[AUTONOMY] YUI проверила состояние, но решила промолчать.", flush=True)

            except requests.exceptions.RequestException as e:
                print(f"[AUTONOMY] Бэкенд недоступен или занят: {e}", flush=True)
            except Exception as e:
                print(f"[AUTONOMY ERROR] {e}", flush=True)
                
            time.sleep(self.check_interval)

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._autonomy_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)