# -*- coding: utf-8 -*-
"""
Управление долговременной памятью: сохранение фактов, поиск (векторный + BM25),
автоматический контекст, извлечение фактов из истории.
"""
import os
import re
import threading
import datetime
from typing import List, Dict

from rank_bm25 import BM25Okapi

from scripts.config import (
    MEMORY_DIR,
    AUTO_CONTEXT_MAX_FILES,
    AUTO_CONTEXT_MAX_LINES_PER_FILE,
    AUTO_CONTEXT_MAX_CHARS,
    VECTOR_DUPLICATE_THRESHOLD,
    DUPLICATE_WORD_OVERLAP,
    VECTOR_SEARCH_THRESHOLD,
    VECTOR_SEARCH_STRONG_THRESHOLD,
    VECTOR_SEARCH_TOPIC_THRESHOLD,
    FACT_CONFIDENCE_DROP_THRESHOLD,
    FACT_CONFIDENCE_ANCHOR_THRESHOLD,
)
from scripts.memory.vector import VectorSearchEngine

# Необязательный маркер доверия к факту в начале строки: (c=-1.00) ... (c=0.90) ...
# confidence по умолчанию 0.0 (теория/предположение), если маркера нет — это
# сохраняет полную обратную совместимость со старыми файлами памяти.
# Смысл шкалы (как в kuni): -1 = опровергнуто/ложь, 0 = теория, 1 = подтверждённая
# истина. LLM никогда не пишет confidence=1 сама — либо 0 по умолчанию, либо явно
# указывает низкое/отрицательное значение при исправлении/опровержении.
_CONFIDENCE_TAG_RE = re.compile(r'^\(c=([+-]?[\d.]+)\)\s*')


def _strip_confidence_tag(text: str) -> tuple:
    """Возвращает (confidence: float, текст без маркера). confidence=0.0, если маркера нет."""
    m = _CONFIDENCE_TAG_RE.match(text)
    if not m:
        return 0.0, text
    try:
        conf = float(m.group(1))
    except ValueError:
        return 0.0, text
    return conf, text[m.end():]


# Метка доверия — и в неряшливом виде, который пишет модель: «(c=-1» без скобки, «c = -1)», «(c= -0.5)»
_FACT_LINE_RE = re.compile(r'^\[(.*?)\]\s*(?:\(?\s*c\s*=\s*([+-]?\d+(?:\.\d+)?)\s*\)?\s*)?(.*)')
# Начало следующего факта посреди строки: «... шлем. [user/interests] Интересуется ...»
_NEXT_FACT_RE = re.compile(r'\s+(?=\[[A-Za-z_]+(?:/[\w-]+)+\])')


def parse_fact_line(line: str):
    """
    Разбирает строку вида `[путь/к/факту] (c=-1) Текст факта.` — общий формат,
    используемый и при обычном извлечении фактов (context.extract_and_save_facts),
    и при сон-консолидации (ReflectionManager). Метка confidence опциональна.
    Возвращает (path, confidence, text) или None, если строка не распознана
    или текст факта пуст.
    """
    match = _FACT_LINE_RE.match(line)
    if not match:
        return None
    path = match.group(1).strip()
    try:
        confidence = float(match.group(2)) if match.group(2) else 0.0
    except ValueError:
        confidence = 0.0
    text = match.group(3).strip()
    if not path or not text:
        return None
    return path, confidence, text


def parse_fact_lines(line: str) -> list:
    """
    Строка вывода модели -> [(path, confidence, text)]. Модель бывает неряшлива (замерено):
    два факта в одной строке («... [user/interests] ...»), путь повторён после метки
    («[user/profile] (c=-1 [user/profile] Зовут Вадим.») — раньше опровержение из-за этого
    сохранялось как новый факт с мусором «(c=-1» в тексте, а не удаляло старый.
    """
    facts, pending = [], None  # pending: (путь, confidence) метки, оставшейся без текста
    for part in _NEXT_FACT_RE.split(line.strip()):
        match = _FACT_LINE_RE.match(part.strip())
        if not match:
            continue
        path, text = match.group(1).strip(), match.group(3).strip()
        confidence = float(match.group(2)) if match.group(2) else None
        if not text:
            pending = (path, confidence) if confidence is not None else pending
            continue
        if confidence is None and pending and pending[0] == path:
            confidence = pending[1]
        pending = None
        if path:
            facts.append((path, confidence if confidence is not None else 0.0, text))
    return facts


# ---------- Слова фактов: основы и «отличительные» слова ----------

STOP_WORDS = {
    'и','в','во','не','что','он','на','я','с','со','как','а','то',
    'все','она','так','его','но','да','ты','к','у','же','вы','за',
    'бы','по','только','ее','мне','было','вот','от','меня','еще','нет',
    'о','из','ему','теперь','когда','даже','ну','вдруг','ли','если',
    'уже','или','ни','быть','был','него','до','вас','нибудь','опять',
    'уж','вам','ведь','там','потом','себя','ничего','ей','может','они',
    'тут','где','есть','надо','ней','для','мы','тебя','их','чем','была',
    'сам','чтоб','без','будто','чего','раз','тоже','себе','под','будет',
    'ж','тогда','кто','этот','того','потому','этого','какой','совсем',
    'ним','здесь','этом','один','почти','мой','тем','чтобы','нее','сейчас',
    'были','куда','зачем','всех','никогда','можно','при','наконец','два',
    'об','другой','хоть','после','над','больше','тот','через','эти','нас',
    'про','всего','них','какая','много','разве','три','эту','моя','впрочем',
    'хорошо','свою','этой','перед','иногда','лучше','чуть','том','нельзя',
    'такой','им','более','всегда','конечно','всю','между',
    # есть почти в каждом факте о пользователе, и слова-время — общими словами не считаются
    # (иначе «посоветуй фильм на вечер» находил «слушает синтвейв по вечерам»)
    'пользователь','пользователя','пользователю','пользователем','пользователе',
    'вечер','вечером','вечерам','вечера','утро','утром','день','днём','ночью','сегодня','завтра',
    'вчера','час','время','какую','каким','каком','сколько','моего','моей','мою','мои',
    'the','and','what','my','your','you','is','are','was','how','who','for','with','that','this',
}
_ENDINGS = sorted(['ями', 'ами', 'ого', 'его', 'ому', 'ему', 'ыми', 'ими', 'иям', 'иях', 'ешь', 'ишь',
                   'ях', 'ах', 'ов', 'ев', 'ей', 'ой', 'ий', 'ый', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие', 'ую',
                   'юю', 'ом', 'ем', 'ам', 'ям', 'их', 'ых', 'ть', 'ет', 'ют', 'ит', 'ат', 'ят', 'им',
                   'а', 'я', 'ы', 'и', 'у', 'ю', 'е', 'о', 'ь'], key=len, reverse=True)


def _stem(word: str) -> str:
    """Грубая основа русского слова: без одного окончания и не длиннее 6 букв («кота» = «кот», «грибами» = «грибы»)."""
    word = word.lower().replace('ё', 'е')
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            word = word[:-len(ending)]
            break
    return word[:6]


def content_stems(text: str) -> set:
    """Основы значимых слов (без стоп-слов и слов короче 3 букв)."""
    return {_stem(w) for w in re.findall(r"[a-zа-яё]+", text.lower().replace('ё', 'е'))
            if len(w) > 2 and w not in STOP_WORDS}


def distinct_tokens(text: str) -> set:
    """Имена (слова с заглавной не в начале), числа и латиница: по ним различаются похожие факты."""
    tokens = set()
    for m in re.finditer(r"[A-Za-zА-Яа-яЁё]+|\d+", text):
        w = m.group(0)
        # Заглавная в начале предложения — не имя («…контентом. Предпочитает…»)
        sentence_start = not text[:m.start()].strip() or text[:m.start()].rstrip()[-1] in '.!?…'
        if w.isdigit() or re.search(r'[A-Za-z]', w) or (w[0].isupper() and not sentence_start):
            tokens.add(w if w.isdigit() else _stem(w))
    return tokens


def same_fact(a: str, b: str) -> bool:
    """
    Один и тот же факт другими словами. Отличающиеся имена/числа/латиница — всегда разные факты
    («Кота зовут Барсик» и «Кота зовут Мурзик», «любит Z.A.T.O.» и «любит Katawa Shoujo»).
    """
    if distinct_tokens(a) != distinct_tokens(b):
        return False
    sa, sb = content_stems(a), content_stems(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / min(len(sa), len(sb)) >= DUPLICATE_WORD_OVERLAP


# Модель при сохранении каждый раз изобретала путь заново, и одна тема расползалась по файлам:
# user/interests и system/user/interests, yui/character и system/yui/yui_character, рефлексии
# в четырёх местах. Канонические пути — ровно те, что читают SoulManager и рефлексия.
_PATH_ALIASES = [
    (r'^system/user/', 'user/'),
    (r'^yui/(?:yui_)?', 'system/yui/yui_'),
    (r'^system/yui/(?!yui_)', 'system/yui/yui_'),
    (r'^system/yui/yui_(?:tastes?|likes|memory_structure)$', 'system/yui/yui_preferences'),
    (r'^system/pc/(?!pc_)', 'system/pc/pc_'),
    (r'^(?:system/)?(?:reflections?|reflective)(?:/.*)?$', 'reflections/reflection_notes'),
]


def canonical_path(path: str) -> str:
    """Приводит путь факта к канонической структуре памяти (см. _PATH_ALIASES)."""
    for pattern, replacement in _PATH_ALIASES:
        path = re.sub(pattern, replacement, path, flags=re.IGNORECASE)
    return path


# Темы, которые e5 не связывает сама: «во что я играю?» не похоже на «любит визуальную новеллу Z.A.T.O.»,
# «что заказать на ужин?» — на «любимая еда — пицца». Слова приводятся к основам через _stem.
TOPIC_WORDS = {
    "игры": "игра игры игру играть играю играет играешь поиграть сыграть игровой геймер гейм "
            "новелла новеллы новеллу шутер рпг rpg квест приставка консоль геймпад steam стим",
    "еда": "еда еду еды ест поесть съесть кушать ужин ужинать обед обедать завтрак перекус голодный "
           "готовить приготовить рецепт блюдо кухня вкусно пицца суп",
    "музыка": "музыка музыку музыки песня песни песню трек треки альбом плейлист слушать слушаю слушает "
              "рок джаз рэп синтвейв",
    "питомцы": "питомец питомцы питомца животное животные кот кота коты котик кошка кошку кошачью собака "
               "собаку пёс хомяк попугай аквариум",
}
_TOPIC_STEMS = {topic: {_stem(w) for w in words.split()} for topic, words in TOPIC_WORDS.items()}


def topics_of(stems: set) -> set:
    """Темы, к которым относятся основы слов."""
    return {topic for topic, topic_stems in _TOPIC_STEMS.items() if stems & topic_stems}


# Вопрос о самих собеседниках, а не общий: «во что я играю?», «что тебе нравится?» —
# или безличный инфинитив, который по-русски тоже про говорящего: «что поесть?», «во что поиграть?».
# Без этого тема цепляла общие вопросы: «сколько живут кошки?» -> «кота зовут Барсик».
_PERSONAL_WORDS = {'я', 'мне', 'меня', 'мной', 'мой', 'моя', 'мое', 'моё', 'мою', 'мои', 'моих', 'моим', 'моей',
                   'моего', 'мы', 'нам', 'нас', 'наш', 'наша', 'наши', 'ты', 'тебе', 'тебя', 'тобой', 'твой',
                   'твоя', 'твою', 'твои', 'твоих', 'твоей', 'твоего', 'i', 'me', 'my', 'we', 'our', 'you', 'your'}


def query_topics(query: str) -> set:
    """Темы запроса — только если он о собеседниках; для общих вопросов пусто."""
    words = re.findall(r"[a-zа-яё]+", query.lower())
    personal = any(w in _PERSONAL_WORDS or w.endswith(('ть', 'ться')) for w in words)
    return topics_of({_stem(w) for w in words}) if personal else set()


def is_relevant(query_stems: set, query_topic_set: set, score: float, fact: str) -> bool:
    """
    Факт подходит запросу: очень похожий по смыслу, или похожий и с общим значимым словом,
    или чуть менее похожий, но на ту же тему (игры, еда...) — см. query_topics.
    """
    if score >= VECTOR_SEARCH_STRONG_THRESHOLD:
        return True
    fact_stems = content_stems(fact)
    if score >= VECTOR_SEARCH_THRESHOLD and query_stems & fact_stems:
        return True
    return score >= VECTOR_SEARCH_TOPIC_THRESHOLD and bool(query_topic_set & topics_of(fact_stems))


class MemoryManager:
    def __init__(self, base_dir: str = MEMORY_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.vector_engine = VectorSearchEngine()
        self._file_lock = threading.Lock()
        # Кэш BM25-индекса, см. _bm25_search: без него КАЖДЫЙ вызов
        # search_memory перечитывает и токенизирует содержимое ВСЕХ файлов
        # памяти заново, хотя обычно ничего не менялось с прошлого поиска.
        self._bm25_cache = None  # (fingerprint, BM25Okapi|None, file_paths, documents)

    # ---------- Вспомогательные методы ----------
    def _validate_path(self, path: str) -> str:
        """Проверяет и санирует путь: макс 2 уровня, удаляет недопустимые символы, обрезает длину."""
        clean_path = path.replace("\\", "/").strip("/")
        if clean_path.endswith(".md"):
            clean_path = clean_path[:-3]

        # Убираем возможный префикс "memory/"
        base_name = os.path.basename(self.base_dir).lower()
        if clean_path.lower().startswith(f"{base_name}/"):
            clean_path = clean_path[len(base_name)+1:]
        elif clean_path.lower() == base_name:
            clean_path = ""

        parts = clean_path.split("/")
        if len(parts) > 3:
            raise ValueError(f"Лимит вложенности превышен (макс 2 уровня), получено: {len(parts)}")

        sanitized_parts = []
        for part in parts:
            part = re.sub(r'[<>:"/\\|?*]', '', part)
            part = part[:25]
            part = part.strip(". ")
            if part:
                sanitized_parts.append(part)

        if not sanitized_parts:
            return "misc/default"

        final_path = canonical_path("/".join(sanitized_parts))
        if len(final_path) > 80:
            final_path = final_path[:80]
        return final_path

    def _tokenize_russian(self, text: str) -> list:
        """Токенизация с удалением стоп-слов для BM25."""
        words = re.findall(r'[a-zа-яё0-9]+', text.lower())
        return [w for w in words if len(w) > 2 and w not in STOP_WORDS]

    def _memory_fingerprint(self) -> tuple:
        """
        Дешёвый отпечаток состояния памяти: (путь, mtime) по каждому .md
        файлу. Позволяет проверить "не изменилось ли что-то с прошлого
        BM25-поиска" за одно перечисление файлов, не читая их содержимое.
        """
        fp = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if not file.endswith(".md"):
                    continue
                fpath = os.path.join(root, file)
                try:
                    fp.append((fpath, os.path.getmtime(fpath)))
                except OSError:
                    continue
        return tuple(sorted(fp))

    def _bm25_search(self, query: str, top_k: int = 3) -> list:
        """
        BM25 поиск по всем .md файлам. Индекс кэшируется в self._bm25_cache и
        перестраивается только когда _memory_fingerprint() реально изменился —
        раньше это перечитывало и токенизировало ВСЕ файлы памяти на каждый
        вызов search_memory, что становится всё дороже по мере роста памяти.
        """
        fingerprint = self._memory_fingerprint()
        if self._bm25_cache is None or self._bm25_cache[0] != fingerprint:
            documents = []
            file_paths = []
            for fpath, _mtime in fingerprint:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        documents.append(content)
                        file_paths.append(os.path.relpath(fpath, self.base_dir).replace("\\", "/"))
                except Exception:
                    continue

            bm25 = None
            if documents:
                tokenized_corpus = [self._tokenize_russian(doc) for doc in documents]
                bm25 = BM25Okapi(tokenized_corpus)
            self._bm25_cache = (fingerprint, bm25, file_paths, documents)

        _fingerprint, bm25, file_paths, documents = self._bm25_cache
        if bm25 is None:
            return []

        tokenized_query = self._tokenize_russian(query)
        if not tokenized_query:
            return []

        scores = bm25.get_scores(tokenized_query)
        scored_results = sorted(zip(file_paths, documents, scores), key=lambda x: x[2], reverse=True)
        valid = [res for res in scored_results if res[2] > 0]
        return valid[:top_k]

    # ---------- Основные методы ----------
    def save_fact(self, path: str, content: str, confidence: float = 0.0) -> str:
        """
        Сохраняет факт в файл памяти с дедупликацией.

        Дубль — это тот же факт другими словами (см. same_fact) с высоким смысловым сходством
        отдельной строки (VECTOR_DUPLICATE_THRESHOLD). Дубль в другом файле — факт не пишется;
        дубль в этом же файле — старая строка заменяется новой. Похожие, но разные факты
        («Барсик рыжий» и «Мурзик чёрный») сохраняются оба.

        :param confidence: доверие к факту, см. _CONFIDENCE_TAG_RE выше.
            confidence <= FACT_CONFIDENCE_DROP_THRESHOLD трактуется как
            РЕТРАКЦИЯ: вместо записи новой строки ищутся и удаляются те же факты
            (это опровержение, а не новый факт) — если только они не являются
            anchor-строками (confidence >= FACT_CONFIDENCE_ANCHOR_THRESHOLD),
            которые сон-консолидация и обычное опровержение не имеют права трогать.
        Возвращает строку статуса.
        """
        try:
            validated = self._validate_path(path)
            full_path = os.path.join(self.base_dir, f"{validated}.md")
            new_fact = content.strip()
            is_retraction = confidence <= FACT_CONFIDENCE_DROP_THRESHOLD

            duplicates = []
            if not is_retraction:
                duplicates = [hit for hit in self.vector_engine.search_lines(new_fact, top_k=5,
                                                                             threshold=VECTOR_DUPLICATE_THRESHOLD)
                              if same_fact(new_fact, hit["line"])]
                if any(hit["id"] != validated for hit in duplicates):
                    return "[MEMORY] ACK. Факт уже существует в памяти под другим путём."
            duplicate_lines = {hit["line"] for hit in duplicates}

            with self._file_lock:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                if is_retraction and not os.path.exists(full_path):
                    return "[MEMORY] Опровержение принято, но похожих фактов не найдено."

                lines_to_write = []
                removed_count = 0
                confirmed_by_anchor = False

                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8") as f:
                        existing_lines = f.readlines()

                    for line in existing_lines:
                        clean_line = line.strip("- ").strip()
                        if not clean_line:
                            continue
                        line_without_ts = re.sub(r'\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', clean_line)
                        old_confidence, old_fact = _strip_confidence_tag(line_without_ts)
                        old_fact = old_fact.strip()
                        is_anchor = old_confidence >= FACT_CONFIDENCE_ANCHOR_THRESHOLD
                        matches = same_fact(new_fact, old_fact) if is_retraction else old_fact in duplicate_lines

                        if matches and is_anchor:
                            # Anchor-факты (подтверждённая истина) неприкосновенны:
                            # ни обычная перезапись, ни ретракция не могут их убрать.
                            confirmed_by_anchor = not is_retraction
                            lines_to_write.append(line)
                        elif matches:
                            removed_count += 1  # старая строка заменяется новой / удаляется ретракцией
                        else:
                            lines_to_write.append(line)

                if confirmed_by_anchor:
                    return "[MEMORY] ACK. Это уже подтверждено в памяти."

                if is_retraction:
                    # Ретракция ничего не добавляет — только удаляет тот же факт.
                    status = (f"[MEMORY] Опровержение принято, удалено строк: {removed_count}."
                              if removed_count else "[MEMORY] Опровержение принято, но похожих фактов не найдено.")
                else:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    conf_tag = f"(c={confidence:+.2f}) " if confidence != 0.0 else ""
                    lines_to_write.append(f"- [{timestamp}] {conf_tag}{new_fact}\n")
                    status = "[MEMORY] OK"

                with open(full_path, "w", encoding="utf-8") as f:
                    f.writelines(lines_to_write)

                # Обновляем векторный индекс для этого файла
                with open(full_path, "r", encoding="utf-8") as f:
                    full_content = f.read().strip()
                if full_content:
                    self.vector_engine.add_document(doc_id=validated, text=full_content)
                elif is_retraction:
                    # Ретракция стёрла последнюю строку — файл опустел, индекс
                    # должен забыть его, иначе останется ссылка на текст,
                    # которого больше нет на диске.
                    self.vector_engine.remove_document(validated)

            return status

        except ValueError as e:
            return f"[MEMORY ERROR] {e}"

    def read_mutable_and_anchor_lines(self, path: str) -> tuple:
        """
        Читает файл памяти по path и делит его строки на:
        - anchors: (confidence, text) с confidence >= FACT_CONFIDENCE_ANCHOR_THRESHOLD —
          подтверждённая истина, неприкосновенна для сон-консолидации.
        - mutable: (confidence, text) всё остальное — можно сжимать, объединять, переписывать.
        Используется ReflectionManager.sleep-консолидацией.
        Возвращает (anchors, mutable) — оба списка кортежей (confidence, text).
        Несуществующий файл — ([], []).
        """
        try:
            validated = self._validate_path(path)
        except ValueError:
            return [], []
        full_path = os.path.join(self.base_dir, f"{validated}.md")
        if not os.path.exists(full_path):
            return [], []

        anchors, mutable = [], []
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                clean_line = line.strip("- ").strip()
                if not clean_line:
                    continue
                without_ts = re.sub(r'^\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', clean_line)
                confidence, text = _strip_confidence_tag(without_ts)
                text = text.strip()
                if not text:
                    continue
                if confidence >= FACT_CONFIDENCE_ANCHOR_THRESHOLD:
                    anchors.append((confidence, text))
                else:
                    mutable.append((confidence, text))
        return anchors, mutable

    def read_fact_records(self, path: str) -> list:
        """Факты файла по порядку: [{"date": "YYYY-MM-DD HH:MM"|None, "confidence": float, "text": str}]."""
        try:
            validated = self._validate_path(path)
        except ValueError:
            return []
        full_path = os.path.join(self.base_dir, f"{validated}.md")
        if not os.path.exists(full_path):
            return []
        records = []
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                clean = line.strip("- ").strip()
                if not clean:
                    continue
                m = re.match(r'^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2})\]\s*', clean)
                confidence, text = _strip_confidence_tag(clean[m.end():] if m else clean)
                if text.strip():
                    records.append({"date": m.group(1) if m else None, "confidence": confidence, "text": text.strip()})
        return records

    def write_fact_records(self, path: str, records: list) -> None:
        """Переписывает файл фактами records (даты сохраняются, без даты — текущая) и обновляет индекс."""
        try:
            validated = self._validate_path(path)
        except ValueError:
            return
        full_path = os.path.join(self.base_dir, f"{validated}.md")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = []
        for r in records:
            conf = r.get("confidence", 0.0)
            conf_tag = f"(c={conf:+.2f}) " if conf != 0.0 else ""
            lines.append(f"- [{r.get('date') or now}] {conf_tag}{r['text'].strip()}\n")
        with self._file_lock:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            content = "".join(lines).strip()
            if content:
                self.vector_engine.add_document(doc_id=validated, text=content)
            else:
                self.vector_engine.remove_document(validated)

    def rewrite_mutable_lines(self, path: str, new_mutable: list) -> None:
        """
        Заменяет ВСЮ изменяемую (не-anchor) часть файла памяти на new_mutable —
        список (confidence, text). Anchor-строки (confidence >= FACT_CONFIDENCE_ANCHOR_THRESHOLD)
        сохраняются как есть, консолидация их не касается. Новые строки получают
        свежий timestamp — как и в kuni, "свежепереписанное" воспоминание снова
        становится "недавним" для будущих циклов сна.
        confidence всегда обрезается ниже anchor-порога: сон не имеет права
        сама назначить факту статус подтверждённой истины.
        Используется ReflectionManager.sleep-консолидацией.
        """
        try:
            validated = self._validate_path(path)
        except ValueError:
            return
        full_path = os.path.join(self.base_dir, f"{validated}.md")

        with self._file_lock:
            anchors = []
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        clean_line = line.strip("- ").strip()
                        if not clean_line:
                            continue
                        without_ts = re.sub(r'^\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', clean_line)
                        confidence, _ = _strip_confidence_tag(without_ts)
                        if confidence >= FACT_CONFIDENCE_ANCHOR_THRESHOLD:
                            anchors.append(line if line.endswith("\n") else line + "\n")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            new_lines = []
            for confidence, text in new_mutable:
                text = text.strip()
                if not text:
                    continue
                if confidence <= FACT_CONFIDENCE_DROP_THRESHOLD:
                    continue  # опровергнуто — промпт сон-консолидации обещает, что такие факты удаляются
                confidence = min(confidence, FACT_CONFIDENCE_ANCHOR_THRESHOLD - 0.01)
                conf_tag = f"(c={confidence:+.2f}) " if confidence != 0.0 else ""
                new_lines.append(f"- [{timestamp}] {conf_tag}{text}\n")

            lines_to_write = anchors + new_lines
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.writelines(lines_to_write)

            with open(full_path, "r", encoding="utf-8") as f:
                full_content = f.read().strip()
            if full_content:
                self.vector_engine.add_document(doc_id=validated, text=full_content)
            else:
                self.vector_engine.remove_document(validated)

    def search_facts(self, query: str, top_k: int = 3) -> str:
        """
        Поиск по памяти. Поддерживает спецкоманды:
        - '__tree__' → структура папок
        - '__all__' → все файлы
        - 'FILE:путь' → содержимое конкретного файла
        Иначе – векторный поиск, при пустом результате – BM25.
        """
        if query.strip() == "__tree__":
            return self._get_tree()

        if query.strip() == "__all__":
            all_content = []
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    if not file.endswith(".md"):
                        continue
                    fpath = os.path.join(root, file)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if content:
                            rel_path = os.path.relpath(fpath, self.base_dir).replace("\\", "/")
                            all_content.append(f"Файл: {rel_path}\n{content}")
                    except Exception:
                        continue
            return "\n\n".join(all_content) if all_content else "Память пуста."

        if query.strip().startswith("FILE:"):
            # Путь приходит из вывода модели (а её могла подговорить прочитанная страница):
            # без проверки "FILE:C:/Users/.../notes" или "../.." читали бы любой .md на диске.
            try:
                file_path = self._validate_path(query.strip()[5:].strip())
            except ValueError as e:
                return f"Недопустимый путь: {e}"
            full_path = os.path.join(self.base_dir, f"{file_path}.md")
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    return f"Содержимое {file_path}.md:\n{content}" if content else "Файл пуст."
                except Exception:
                    return "Ошибка чтения файла."
            return "Файл не найден."

        # Векторный поиск по отдельным фактам: показываем только подошедшие факты
        query_stems, query_topic_set = content_stems(query), query_topics(query)
        output = []
        for res in self.vector_engine.search(query, top_k=top_k * 3, threshold=VECTOR_SEARCH_TOPIC_THRESHOLD):
            facts = [fact for score, fact in res.get("hits", []) if is_relevant(query_stems, query_topic_set, score, fact)]
            if facts:
                listed = "\n".join(f"- {f}" for f in facts[:5])
                output.append(f"Файл: {res['id']} (Сходство: {res['score']:.2f})\n{listed}\n")
            if len(output) >= top_k:
                break
        if output:
            return "\n".join(output)

        # Fallback: BM25
        bm25_results = self._bm25_search(query, top_k=top_k)
        if bm25_results:
            output = []
            for fpath, content, score in bm25_results:
                if len(content) > 300:
                    content = content[:300] + "\n[...ОБРЕЗАНО]"
                output.append(f"Файл: {fpath} (BM25)\n{content}\n")
            return "\n".join(output)

        return "По памяти ничего не найдено."

    def _get_tree(self) -> str:
        """Генерирует текстовое дерево директорий памяти."""
        tree_str = "/memory\n"
        for root, dirs, files in os.walk(self.base_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if f.endswith('.md')]
            level = root.replace(self.base_dir, '').count(os.sep)
            indent = '  ' * level
            tree_str += f'{indent}{os.path.basename(root)}/\n'
            subindent = '  ' * (level + 1)
            for file in files:
                tree_str += f'{subindent}{file}\n'
        return tree_str if len(tree_str) > 8 else "Память абсолютно пуста."

    def get_auto_context(self, query: str, max_files: int = AUTO_CONTEXT_MAX_FILES,
                         max_lines: int = AUTO_CONTEXT_MAX_LINES_PER_FILE,
                         max_chars: int = AUTO_CONTEXT_MAX_CHARS, allowed=None) -> str:
        """
        Факты из памяти, подходящие к реплике, — для автоматической инъекции в контекст каждого хода.
        Ничего не подошло — пустая строка: лучше ничего, чем случайный факт (раньше на любую реплику
        подмешивалась последняя строка найденного файла, и на «как дела?» Юи видела «мама живёт в Казани»).
        """
        query_stems, query_topic_set = content_stems(query), query_topics(query)
        context_lines, seen, total_len, files_used = [], set(), 0, 0
        for res in self.vector_engine.search(query, top_k=max_files * 3, threshold=VECTOR_SEARCH_TOPIC_THRESHOLD):
            taken = 0
            for score, fact in res.get("hits", []):
                if taken >= max_lines or not is_relevant(query_stems, query_topic_set, score, fact):
                    continue
                if allowed is not None and not allowed(res["id"], fact):
                    continue  # гостю из Telegram — только его файл и черты Юи, не память о владельце
                norm = re.sub(r'[^a-zа-яё0-9]', '', fact.lower())
                if not norm or norm in seen:
                    continue
                to_add = f"- {fact.rstrip('.')}."
                if total_len + len(to_add) > max_chars:
                    return "\n".join(context_lines)
                seen.add(norm)
                context_lines.append(to_add)
                total_len += len(to_add)
                taken += 1
            files_used += bool(taken)
            if files_used >= max_files:
                break
        return "\n".join(context_lines)

    def get_fact_extraction_prompt(self, history_to_compress: List[Dict[str, str]], tree_str: str = "") -> str:
        """
        Генерирует промпт для извлечения фактов из сжимаемой истории.
        """
        hist_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history_to_compress])
        tree_context = f"Текущая структура памяти:\n{tree_str}\n" if tree_str else self._get_tree()

        return f"""Извлеки ТОЛЬКО важные долгосрочные факты.
{tree_context}
ЖЕСТКИЕ ПРАВИЛА ФОРМАТИРОВАНИЯ:
1. ЗАПРЕЩЕНО писать рассуждения, анализ, слова "Анализ", "Шаг". ПИШИ ТОЛЬКО ИТОГОВЫЕ ФАКТЫ.
2. Формат строго построчный: [категория/тема] Текст факта.
3. ПРАВИЛА КАТЕГОРИЙ (ВЫПОЛНЯТЬ СТРОГО):
   - СНАЧАЛА ищи подходящий файл в структуре памяти выше и пиши туда. Новый файл — только если
     подходящего правда нет. Не создавай синонимы существующих файлов (tastes при наличии preferences).
   - Всё о пользователе — его вкусы, интересы, привычки, люди вокруг, настроение — [user/...]:
     [user/interests] <что любит>. [user/profile] <имя, если он его назвал>. [user/family] <кто из близких и как зовут>.
     (Это ШАБЛОНЫ формата, а не факты: пиши только то, что правда было в истории.)
   - О тебе самой: черты характера -> [system/yui/yui_character], что нравится/не нравится -> [system/yui/yui_preferences].
   - О ПК, железе -> [system/pc/pc_...].
   - Знания о мире, не о пользователе и не о тебе -> своя тема: [<тема>/<подтема>] <факт>.
4. Один факт — одна строка. Не повторяй то, что уже есть в памяти другими словами.
5. Если фактов нет, пиши только: NULL
6. ОПРОВЕРЖЕНИЕ: если из истории ясно, что пользователь ИСПРАВИЛ тебя, и что-то,
   во что ты раньше верила, ОКАЗАЛОСЬ НЕПРАВДОЙ — не пиши новый факт поверх старого,
   а ЯВНО ОПРОВЕРГНИ его меткой (c=-1) сразу после категории:
   [user/dislikes] (c=-1) Не любит собак.
   Это УДАЛИТ противоречащую запись из памяти вместо того, чтобы обе версии лежали
   рядом. Используй ТОЛЬКО когда действительно есть прямое опровержение, не для
   обычных новых фактов (у обычных фактов метки быть не должно).
7. ДНЕВНИК: последней строкой — что ТЫ почувствовала в этом разговоре, от первого лица, с причиной,
   одно-два предложения: [diary/{datetime.date.today():%Y-%m-%d}] <чувство> — потому что <что произошло>.
   Честно, без дежурной радости; разговор был пустой или ничего не задел — дневник не пиши.

История:
{hist_str}

Факты:"""