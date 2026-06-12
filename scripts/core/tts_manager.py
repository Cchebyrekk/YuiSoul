import pygame
import tempfile
import os
import threading
import queue
import re
import subprocess
import requests

class TTSManager:
    def __init__(self, piper_bin="tts/piper.exe", piper_model="tts/ru_RU-ruslan-medium.onnx", rvc_api_url="http://127.0.0.1:7865", rvc_model_name="yui_model.pth"):
        self.piper_bin = piper_bin
        self.piper_model = piper_model
        self.piper_config = f"{piper_model}.json"
        self.rvc_api_url = rvc_api_url
        self.rvc_model_name = rvc_model_name
        
        self.audio_queue = queue.Queue()
        self._running = True
        
        # Проверка Piper
        if not os.path.exists(self.piper_bin) or not os.path.exists(self.piper_model):
            print("[TTS FATAL] piper.exe или модель не найдены в папке tts/. Голос отключен.")
            self._running = False
            return
            
        # Инициализация аудио микшера
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=2048)
        except Exception as e:
            print(f"[TTS FATAL] Pygame mixer init error: {e}")
            self._running = False
            return
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str):
        if not self._running or not text: return
        
        # Санитария XML/Markdown
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[\*\_\#\`\[\]\(\)]', '', text).strip()
        
        if text:
            self.audio_queue.put(text)

    def _worker(self):
        while self._running:
            try:
                text = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            base_path = ""
            rvc_path = ""
            try:
                # ШАГ 1: Генерация базы через Piper (Офлайн, мгновенно, 0 VRAM)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_base:
                    base_path = tmp_base.name
                
                process = subprocess.Popen(
                    [self.piper_bin, "--model", self.piper_model, "--config", self.piper_config, "--output_file", base_path],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                _, stderr = process.communicate(input=text.encode('utf-8'))
                
                if process.returncode != 0:
                    print(f"[TTS ERROR] Piper упал: {stderr.decode('utf-8', errors='ignore')}")
                    continue
                
                # ШАГ 2: Отправка на локальный RVC сервер (Офлайн)
                rvc_success = False
                try:
                    with open(base_path, 'rb') as f:
                        files = {'audio_file': ('base.wav', f, 'audio/wav')}
                        data = {
                            'model_name': self.rvc_model_name,
                            'f0_up_key': 0,       # Сдвиг тона (0 = без сдвига)
                            'index_rate': 0.5,     # Влияние индекса (0-1)
                            'device': 'cpu'        # КРИТИЧЕСКИ ВАЖНО: CPU, чтобы не крашнуть llama.cpp
                        }
                        response = requests.post(f"{self.rvc_api_url}/api/infer", files=files, data=data, timeout=30)
                    
                    if response.status_code == 200:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_rvc:
                            rvc_path = tmp_rvc.name
                            tmp_rvc.write(response.content)
                        rvc_success = True
                    else:
                        print(f"[TTS WARNING] RVC API ошибка: {response.status_code}. Фолбэк на Piper.")
                except requests.exceptions.RequestException:
                    print("[TTS WARNING] RVC сервер недоступен. Фолбэк на чистый Piper.")
                
                # ШАГ 3: Воспроизведение
                play_path = rvc_path if rvc_success else base_path
                
                pygame.mixer.music.load(play_path)
                pygame.mixer.music.play()
                
                # Блокировка до конца воспроизведения
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(30)
                    
            except Exception as e:
                print(f"[TTS ERROR] Полный сбой пайплайна: {e}")
            finally:
                self.audio_queue.task_done()
                # Уборка временных файлов
                if base_path and os.path.exists(base_path):
                    try: os.remove(base_path)
                    except: pass
                if rvc_path and os.path.exists(rvc_path):
                    try: os.remove(rvc_path)
                    except: pass

    def stop(self):
        self._running = False
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        self._thread.join(timeout=2)