# stress_test_qwen.py
import requests
import time
import subprocess
import sys

LLAMA_API = "http://127.0.0.1:8080"
TIMEOUT_PREFILL_100K = 300 # 5 минут на первый токен. Если не успел — считаем свопингом.

def get_vram_usage() -> tuple[int, int]:
    """Возвращает (использовано MB, всего MB)"""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding='utf-8', timeout=2
        )
        if res.returncode == 0:
            used, total = map(int, res.stdout.strip().split(","))
            return used, total
    except Exception:
        pass
    return -1, -1

def generate_payload(target_chars: int) -> str:
    """Генерирует мусорную строку заданного размера. 
    Для Qwen/Russian ~1 токен = 3-4 символа. 400к символов ~ 100k токенов."""
    base_chunk = "ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ. Тестовая последовательность для проверки лимитов контекста модели. "
    repeats = (target_chars // len(base_chunk)) + 1
    return (base_chunk * repeats)[:target_chars]

def test_baseline():
    """Тест 1: Короткий контекст (проверка базовой работоспособности)"""
    print("\n[TEST 1] Базовый тест (короткий контекст)...")
    vram_before, vram_total = get_vram_usage()
    
    payload = "Напиши слово 'ОК'."
    start = time.time()
    
    try:
        r = requests.post(f"{LLAMA_API}/completion", json={
            "prompt": f"<start_of_turn>user\n{payload}<end_of_turn>\n<start_of_turn>model\n",
            "n_predict": 5,
            "temperature": 0.0
        }, timeout=30.0)
        
        elapsed = time.time() - start
        vram_after, _ = get_vram_usage()
        
        if r.status_code == 200:
            timings = r.json().get("timings", {})
            tps = timings.get("predicted_per_second", 0)
            print(f"  [OK] Статус: Успешно")
            print(f"  [VRAM] До: {vram_before} MB | После: {vram_after} MB (Дельта: {vram_after - vram_before} MB)")
            print(f"  [SPEED] Промпт: {timings.get('prompt_per_second', 0):.2f} t/s | Генерация: {tps:.2f} t/s")
        else:
            print(f"  [FAIL] Код: {r.status_code} | Ответ: {r.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("  [FAIL] Таймаут базового запроса. Бэкенд завис.")
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")

def test_100k_context():
    """Тест 2: Переполнение 100k токенов"""
    print("\n[TEST 2] Стресс-тест 100k контекста...")
    print("  [INFO] Генерация полезной нагрузки (~400 000 символов)...")
    
    massive_text = generate_payload(200000)
    actual_prompt = f"<start_of_turn>user\n{massive_text}\nВопрос: Какой это текст?<end_of_turn>\n<start_of_turn>model\n"
    
    vram_before, vram_total = get_vram_usage()
    print(f"  [VRAM] Старт: {vram_before} / {vram_total} MB")
    print(f"  [INFO] Отправка запроса. Ожидание первого токена (лимит {TIMEOUT_PREFILL_100K} сек)...")
    
    start = time.time()
    try:
        r = requests.post(f"{LLAMA_API}/completion", json={
            "prompt": actual_prompt,
            "n_predict": 10,
            "temperature": 0.0
        }, timeout=TIMEOUT_PREFILL_100K)
        
        elapsed = time.time() - start
        vram_after, _ = get_vram_usage()
        
        if r.status_code == 200:
            timings = r.json().get("timings", {})
            prompt_eval = timings.get("prompt_ms", 0) / 1000.0
            tps_prompt = timings.get("prompt_per_second", 0)
            tps_gen = timings.get("predicted_per_second", 0)
            
            print(f"  [OK] Запрос выполнен (или частично выполнен)")
            print(f"  [TIME] Время до первого токена: {elapsed:.2f} сек.")
            print(f"  [VRAM] Пик: {vram_after} MB (Дельта: +{vram_after - vram_before} MB)")
            print(f"  [SPEED] Prefill: {tps_prompt:.2f} t/s | Генерация: {tps_gen:.2f} t/s")
            
            if tps_prompt < 5.0:
                print("  [VERDICT] КРИТИЧЕСКОЕ СВАПИРОВАНИЕ. Скорость препроцессинга ниже 5 t/s. Работа невозможна.")
            elif vram_after > (vram_total - 500):
                print("  [VERDICT] ПОРОГОВЫЙ OOM. VRAM заполнен под завязку. Любой следующий запрос уронит процесс.")
            else:
                print("  [VERDICT] УСПЕХ. Железо потянуло 100k (неожиданно, проверь реальное кол-во токенов в логах сервера).")
        else:
            err_text = r.text[:300]
            print(f"  [FAIL] Сервер упал. Код: {r.status_code}")
            print(f"  [ERROR] {err_text}")
            if "exceed_context_size" in err_text or "OOM" in err_text.upper():
                print("  [VERDICT] ОЖИДАЕМЫЙ КРАШ. Физический предел памяти превышен.")
                
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        vram_after, _ = get_vram_usage()
        print(f"  [FAIL] ТАЙМАУТ через {elapsed:.2f} сек.")
        print(f"  [VRAM] Текущий: {vram_after} MB")
        print("  [VERDICT] БЭКЕНД ЗАВИС НА SWAP. DDR4 не справляется с объемом KV-кэша.")
    except requests.exceptions.ConnectionError:
        print("  [FAIL] СОЕДИНЕНИЕ РАЗОРВАНО.")
        print("  [VERDICT] llama.cpp МАССИВНО УПАЛА (SEGFAULT / OOM Killer).")
    except Exception as e:
        print(f"  [FAIL] Неожиданная ошибка: {e}")

if __name__ == "__main__":
    print("="*60)
    print("STRESS TEST: Qwen3.6-35B-A3B + 100k Context")
    print("="*60)
    
    try:
        requests.get(f"{LLAMA_API}/health", timeout=2)
    except Exception:
        print("[CRITICAL] Бэкенд llama.cpp недоступен. Запусти server.exe и повтори.")
        sys.exit(1)
        
    test_baseline()
    test_100k_context()
    print("\n[TOTAL] Тестирование завершено.")