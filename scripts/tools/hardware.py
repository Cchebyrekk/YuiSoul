# -*- coding: utf-8 -*-
"""
Мгновенные показания железа (GPU через nvidia-smi) — по запросу Юи через инструмент check_hardware.
Раньше строка подставлялась в каждый ход, и Юи сохраняла её в память как «факт».
"""
import subprocess


def get_hardware_status() -> str:
    """Возвращает строку с текущим состоянием GPU (температура, загрузка, VRAM)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding='utf-8', timeout=5
        )
        if result.returncode == 0:
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            gpus = []
            for i, line in enumerate(lines):
                temp, util, mem_used, mem_total = map(str.strip, line.split(","))
                gpus.append(f"GPU {i}: {temp}°C | Load: {util}% | VRAM: {mem_used}/{mem_total} MB")
            if gpus:
                return "\n".join(gpus) + "\n(Мгновенные показания — не сохраняй их в память.)"
    except Exception:
        pass
    return "GPU: Данные недоступны"
