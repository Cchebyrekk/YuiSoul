# -*- coding: utf-8 -*-
"""
Единый конфигурационный файл для YUI.
Все настройки вынесены сюда. Изменяйте значения здесь — код останется неизменным.
"""

import os
import torch

# ==================== ПУТИ ====================

# Корень проекта (папка yuisoul)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Папки данных
MODELS_DIR = os.path.join(BASE_DIR, "models")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
TTS_DIR = os.path.join(BASE_DIR, "tts")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
TURBOQUANT_DIR = os.path.join(BASE_DIR, "llama_things", "turboquant-new", "build", "bin")

# LLM модель (путь к .gguf)
LLM_MODEL_PATH = os.path.join(MODELS_DIR, "Qwen3.6-35B-A3B-Uncensored-Genesis-MTP-APEX-Compact.gguf")
LLM_HOST = "127.0.0.1"
LLM_PORT = 8080
LLM_API_URL = f"http://{LLM_HOST}:{LLM_PORT}/v1/chat/completions"

# Векторная память (SentenceTransformer)
VECTOR_MODEL_NAME = "intfloat/multilingual-e5-base"
VECTOR_DEVICE = "cpu"

# Сессии
SESSION_FILE = os.path.join(SESSIONS_DIR, "latest.json")

# ==================== ПАРАМЕТРЫ LLM ====================

# max_tokens включает и рассуждения модели (reasoning), а не только ответ:
# при 256 Qwen3.6 успевала только подумать и обрывалась без ответа (finish=length).

# Параметры для быстрых ответов (приветствия, простые реплики)
FAST_TEMPERATURE = 0.5             # 0.1 делала реплики плоскими и одинаковыми — личности нужна вариативность
FAST_MAX_TOKENS = 1024

# Параметры для сложных размышлений
DEEP_TEMPERATURE = 0.6
DEEP_MAX_TOKENS = 2048

DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.95
DEFAULT_REPEAT_PENALTY = 1.1

# Токены остановки (для прекращения генерации)
STOP_TOKENS = ["<|im_end|>"]

# ==================== АГЕНТ И ЦИКЛ ====================

MAX_STEPS = 10                     # максимум итераций агента
MAX_RETRIES = 3                    # повторные попытки при сбое LLM

# Ограничения по контексту
MAX_CONTEXT_CHARS = 60000          # для сжатия
CONTEXT_TAIL_RATIO = 0.25          # доля хвоста при сжатии

# Таймауты (секунды)
LLM_TIMEOUT = 120.0
HTTP_READ_TIMEOUT = 120.0

# Флаги
ENABLE_AUTONOMY = True            # автономные мысли в фоне
# Простой считается от последнего обращения к Юи (реплика пользователя или конец её ответа),
# а не по клавиатуре/мыши: за компьютером можно работать, главное — не говорить с ней.
AUTONOMY_POLL_INTERVAL = 15        # как часто фоновый поток проверяет условия (сек)
AUTONOMY_IDLE_THRESHOLD = 180      # сколько секунд без обращений к Юи нужно для спонтанной мысли
AUTONOMY_MIN_INTERVAL = 600        # минимум секунд между мыслями; без ответа пользователя растёт: x2, x3...
AUTONOMY_MAX_STEPS = 6            # шагов внутреннего хода (мысль -> поиск -> мысль -> save_memory ...)
# Размышляя сама, Юи может искать, писать в память, смотреть на экран и говорить вслух (speak_aloud).
# Управление ПК (печать, горячие клавиши, запуск/закрытие программ) — только если разрешено:
# иначе она может начать печатать в окно, где пользователь сейчас работает.
AUTONOMY_ALLOW_PC_CONTROL = False

# Право на тишину (по образцу processIgnoreChance из kuni): с этой
# вероятностью перед каждым обращением к LLM ей мягко напоминают, что она не
# ОБЯЗАНА отвечать прямо сейчас — можно позвать инструмент stay_silent и
# промолчать. Это подсказка, а не принуждение: агент решает сам.
# 0.0 полностью отключает — агент будет отвечать на каждое сообщение, как раньше.
SILENCE_NUDGE_CHANCE = 0.15

# Своя воля (scripts/agent/will.py): перед ответом Юи отдельным коротким вызовом
# (без рассуждений, ~0.5-2 с) решает, хочет ли она выполнять просьбу: ДА / НЕТ / ЧАСТИЧНО.
# Нужна потому, что uncensored-модель почти не отказывает сама по ходу ответа.
WILL_CHECK_ENABLED = True

# Рассуждения модели (reasoning) — главный источник задержки голосового ответа: на обычную реплику
# Qwen рассуждала 550-1000 символов, и первая фраза звучала через 9-13 с вместо ~1.3 с.
# True: рассуждения и проверка воли — только для задач (найди/напиши/открой..., см. DEEP_KEYWORDS_RE
# в loop.py) и во внутренних размышлениях; обычный разговор — сразу ответ.
# Переключение режима между запросами KV-кэш не ломает (замерено: ~1 с в обе стороны).
THINK_ON_TASKS_ONLY = True

# Извлечение фактов из диалога занимает единственный слот llama-server на ~14 с. Запускаем его
# только после паузы в разговоре и обрываем, если пользователь заговорил, — иначе быстрая
# реплика ждала бы его в очереди (замерено: 16 с до первого звука вместо ~3 с).
FACT_EXTRACTION_IDLE_DELAY = 15
WILL_CHECK_MAX_TOKENS = 80
WILL_CHECK_TEMPERATURE = 0.7

# ==================== ЗРЕНИЕ ====================

# Инструменты look_at_screen / view_image / zoom_image. Требуют llama-server с --mmproj (vision-проектор
# Qwen3.6); без него сервер отклонит запрос с картинкой — тогда выключи.
VISION_ENABLED = True
SCREENSHOT_MAX_SIDE = 1280         # длинная сторона скриншота после уменьшения (~1000 токенов)
SCREENSHOT_JPEG_QUALITY = 85
ZOOM_MAX_UPSCALE = 3.0             # во сколько раз максимум увеличивать мелкий кроп (мелкий текст)
VISION_MAX_IMAGES_PER_TURN = 3     # сколько последних картинок держать в контексте внутри хода (серия zoom)

# ==================== ИНТЕРНЕТ ====================

# Инструменты search_web (DuckDuckGo через пакет ddgs, без API-ключа) и read_webpage.
WEB_ENABLED = True
WEB_SEARCH_REGION = "ru-ru"        # регион выдачи DuckDuckGo (wt-wt — без привязки к стране)
WEB_SEARCH_MAX_RESULTS = 5
WEB_PAGE_MAX_CHARS = 6000          # сколько текста страницы отдавать модели (~2000 токенов)
WEB_TIMEOUT = 15                   # секунд на поиск/загрузку страницы

# ==================== RAG 2.0 / РЕФЛЕКСИЯ ====================

ENABLE_REFLECTION = True              # фоновая консолидация памяти (ReflectionManager)
REFLECTION_POLL_INTERVAL = 30          # как часто фоновый поток проверяет условия (сек)
REFLECTION_IDLE_THRESHOLD = 300        # сколько секунд без обращений к Юи нужно для рефлексии
REFLECTION_MIN_INTERVAL = 1200         # минимум секунд между циклами рефлексии (20 мин)
REFLECTION_MAX_STEPS = 8              # шагов внутреннего хода рефлексии
REFLECTION_MIN_FACTS = 5               # минимум фактов в памяти, чтобы рефлексия имела смысл
REFLECTION_RECENT_FACTS_WINDOW = 20    # сколько последних фактов анализировать за цикл

# --- Сон-консолидация памяти (по образцу sleepingConsolidation из kuni) ---
# Раз в несколько циклов рефлексии (вместо/вместе с лёгкой "рефлексией-инсайтом")
# YUI перечитывает один файл памяти, находит похожие через векторный поиск,
# и просит LLM сжать/объединить/переписать дубли — как человек, который во
# сне переупаковывает воспоминания за день.
ENABLE_SLEEP_CONSOLIDATION = False
SLEEP_CONSOLIDATION_EVERY_N_CYCLES = 3     # раз в N срабатываний фоновой рефлексии
SLEEP_CONSOLIDATION_RELATED_FILES = 3      # сколько похожих файлов подмешивать к цели
SLEEP_CONSOLIDATION_RECENT_BIAS = 0.8      # шанс выбрать САМЫЙ свежий файл, а не случайный
SLEEP_CONSOLIDATION_MAX_TOKENS = 1024
SLEEP_CONSOLIDATION_TEMPERATURE = 0.3

# Confidence факта: -1 = опровергнуто/ложь, 0 = теория/предположение (по
# умолчанию), 1 = подтверждённая истина. LLM никогда не присваивает 1 сама —
# только 0 по умолчанию либо явную отметку при исправлении/опровержении.
# Ниже этого порога факт не записывается, а вместо этого используется как
# сигнал "удали похожую старую строку" (ретракция), см. MemoryManager.save_fact.
FACT_CONFIDENCE_DROP_THRESHOLD = -0.5
# Confidence выше этого порога (уже сохранённые вручную/системой "истины")
# сон-консолидация не имеет права переписывать или удалять.
FACT_CONFIDENCE_ANCHOR_THRESHOLD = 0.999

# Насколько сильно свежесть факта поднимает его в выдаче поиска (0 = выключено).
# Итоговый скор = cosine_score + RECENCY_BOOST_WEIGHT * recency_factor,
# где recency_factor затухает экспоненциально с периодом полураспада ниже.
# Порог релевантности (VECTOR_SEARCH_THRESHOLD) всегда проверяется по
# «чистому» cosine_score, чтобы свежий мусор не проходил мимо фильтра.
RECENCY_BOOST_WEIGHT = 0.05
RECENCY_HALFLIFE_DAYS = 30

# ==================== TTS ====================

# (OMNIVOICE)

# Параметры модели
OMNIVOICE_LANGUAGE = "ru"                     # язык синтеза
OMNIVOICE_SPEED = 1.0                         # скорость (0.1-5.0)
OMNIVOICE_INSTRUCT = "female, young adult, moderate pitch"   # конструктор голоса
OMNIVOICE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # или "cpu"
OMNIVOICE_DTYPE = "float16" if OMNIVOICE_DEVICE == "cuda" else "float32"

# Путь для кэширования модели (если нужно)
# OMNIVOICE_CACHE_DIR = os.path.join(BASE_DIR, "cache", "omnivoice")
# Если не задан, модель загрузится в стандартную папку кэша

# Дополнительно: можно задать отдельные атрибуты для удобства
OMNIVOICE_ATTRIBUTES = {
    "gender": "female",          # male / female
    "age": "young adult",        # child / teenager / young adult / middle-aged / elderly
    "pitch": "moderate pitch",   # very low / low / moderate / high / very high
    "style": None,               # whisper (опционально)
    "accent": None,              # american accent, british accent, ... (только для английского)
    "dialect": None,             # 四川话 и т.п. (только для китайского)
}

# Silero принимает ТОЛЬКО одно из: 8000, 24000, 48000.
# Раньше частота для apply_tts() не передавалась явно и угадывалась через
# getattr(model, 'sample_rate', 24000), из-за чего реальная частота синтеза
# (48000) не совпадала с частотой, на которой инициализировался pygame.mixer
# и писался WAV (24000) — отсюда "замедленный" низкий голос при воспроизведении.
# Теперь это единственный источник истины: используется и в apply_tts(), и в
# pygame.mixer.init(), и в sf.write().
SILERO_SAMPLE_RATE = 48000
TTS_SPEAKER_RU = "xenia"
# Английский голос (модель scripts/speech/silero_model_en.pt, v3_en). en_21 — по высоте тона
# ближе всего к xenia (~236 Гц); другие женские: en_36, en_5, en_16, en_11. Без файла модели
# английские фразы не озвучиваются.
TTS_SPEAKER_EN = "en_21"
TTS_CHANNELS = 1
TTS_BUFFER = 2048

# ==================== STT ====================

# (Whisper)
WHISPER_MODEL_SIZE = "small"          # tiny, base, small, medium, large
WHISPER_DEVICE = "cpu"                # или "cuda" если хватит VRAM
WHISPER_COMPUTE_TYPE = "int8"         # int8, int16, float32

# Микрофон: часть имени устройства (без учёта регистра) или None — микрофон Windows по умолчанию.
# По умолчанию Windows выбрала микрофон геймпада (Wireless Controller), который отдаёт тишину.
STT_INPUT_DEVICE = "Fifine"

STT_SAMPLERATE = 16000
STT_BLOCK_DURATION = 0.3           # секунд на блок аудио
STT_SILENCE_BLOCKS = 5             # тихих блоков для остановки записи (5 x 0.3 = 1.5 с; было 9 = 2.7 с ожидания после каждой фразы)
STT_LANGUAGE = "ru"                # язык по умолчанию (с него начинаем и на нём остаёмся, если не уверены)
# Языки, на которых можно говорить с Юи. Если их больше одного, язык каждой фразы определяет
# маленький Whisper tiny (~0.3 с) — автоопределение самим small стоило бы +1.4 с на фразу.
# Язык "липкий": переключается, только если tiny уверен (короткое "да" он путает).
STT_LANGUAGES = ("ru", "en")
STT_LANGUAGE_DETECT_MODEL = "tiny"
STT_LANGUAGE_MIN_CONFIDENCE = 0.5
STT_VOLUME_THRESHOLD = 0.015       # порог громкости
STT_MIN_AUDIO_LENGTH = 0.5         # минимальная длина звука во фразе, без хвоста тишины (сек): отсекает щелчки, пропускает "да"

# ==================== ПАМЯТЬ (RAG) ====================

# Лимиты для поиска и инъекции
AUTO_CONTEXT_MAX_FILES = 3
AUTO_CONTEXT_MAX_LINES_PER_FILE = 2
AUTO_CONTEXT_MAX_CHARS = 400

# Пороги сходства для дедупликации
VECTOR_DUPLICATE_THRESHOLD = 0.90   # косинус > 0.90 – дубль
JACCARD_DUPLICATE_THRESHOLD = 0.4   # Жаккард >= 0.4 – дубль внутри файла
VECTOR_SEARCH_THRESHOLD = 0.75      # минимальный скор для выдачи результатов

# ==================== ЭМОЦИОНАЛЬНЫЙ КАНАЛ ====================

EMOTION_WEBSOCKET_URL = "ws://127.0.0.1:8765"  # заглушка для аватара
EMOTION_EXTRACTION_ENABLED = False   # пока отключено

# ==================== ЛОГИРОВАНИЕ ====================

LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR
LOG_LATENCY = True                 # замерять задержки

# ==================== ПРОЧЕЕ ====================

# Потоки для работы с памятью
MEMORY_IO_THREADS = 1               # количество потоков для записи/чтения (для блокировок)

