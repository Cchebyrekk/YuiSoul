import mss
import mss.tools
from PIL import Image
import io
import base64

class VisionCapture:
    def __init__(self, target_size=(1024, 1024)):
        self.target_size = target_size
        self.sct = mss.mss()

    def grab_screen(self, region=None) -> str:
        """
        Захватывает экран, ресайзит, кодирует в base64.
        Возвращает Data URI для отправки в llama.cpp.
        """
        # Если region передан, захватываем только область (для "зума")
        monitor = region if region else self.sct.monitors[1] # 1 - основной монитор в Windows
        
        # Захват raw RGBA
        screenshot = self.sct.grab(monitor)
        
        # Конвертация в PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # Агрессивный даунскейл для экономии контекста
        img = img.resize(self.target_size, Image.LANCZOS)
        
        # Кодировка в base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85) # JPEG меньше весит, чем PNG
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/jpeg;base64,{img_b64}"