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
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
    millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
    return millis // 1000

class AutonomyManager:
    def __init__(self, check_interval: int = 600, idle_threshold: int = 300):
        self.check_interval = check_interval
        self.idle_threshold = idle_threshold
        self._running = False
        self._thread = None
        self.soul_manager = SoulManager()

    def _autonomy_loop(self):
        print("[AUTONOMY] Фоновый процесс инициализирован. Ожидание простоя...")
        while self._running:
            time.sleep(self.check_interval)
            
            if not self._running: break

            idle_sec = get_idle_seconds()
            if idle_sec < self.idle_threshold:
                continue

            idle_min = idle_sec // 60
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            soul_patch = self.soul_manager.generate_soul_patch()
            if not soul_patch:
                soul_patch = "<user_profile>Нет данных о пользователе.</user_profile>"

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Текущее время: {time_str}. Пользователь неактивен {idle_min} минут.\n"
                        f"{soul_patch}\n"
                        f"У тебя есть спонтанное желание высказаться или подумать вслух в одиночестве? "
                        f"Если да — напиши короткую мысль в тегах <thought> и <output>. Если нет — пиши только NULL."
                    )
                }
            ]

            try:
                response = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.8,
                }, timeout=30.0)

                if response.status_code != 200:
                    continue

                raw_reply = response.json()["choices"][0]["message"]["content"].strip()
                
                if not raw_reply or raw_reply == "NULL":
                    continue

                raw_reply = re.sub(r'<thought>.*?</thought>', '', raw_reply, flags=re.DOTALL).strip()
                raw_reply = raw_reply.replace("<output>", "").replace("</output>", "").strip()

                if len(raw_reply) > 10:
                    print(f"\n[YUI AUTONOMOUS] ({idle_min} мин. тишины) {raw_reply}\n")

            except requests.exceptions.RequestException:
                pass
            except Exception as e:
                print(f"[AUTONOMY ERROR] {e}")

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._autonomy_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)