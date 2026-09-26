# -*- coding: utf-8 -*-
"""
Управление долговременной памятью: сохранение фактов, поиск (векторный + BM25),
автоматический контекст, извлечение фактов из истории.
"""
import os
import re
import json
import threading
import datetime
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from scripts.config import (
    MEMORY_DIR,
    AUTO_CONTEXT_MAX_FILES,
    AUTO_CONTEXT_MAX_LINES_PER_FILE,
    AUTO_CONTEXT_MAX_CHARS,
    VECTOR_DUPLICATE_THRESHOLD,
    JACCARD_DUPLICATE_THRESHOLD,
    VECTOR_SEARCH_THRESHOLD,
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


_FACT_LINE_RE = re.compile(r'^\[(.*?)\]\s*(?:\(c=([+-]?[\d.]+)\)\s*)?(.*)')


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

        final_path = "/".join(sanitized_parts)
        if len(final_path) > 80:
            final_path = final_path[:80]
        return final_path

    def _tokenize_russian(self, text: str) -> list:
        """Токенизация с удалением стоп-слов для BM25."""
        words = re.findall(r'[a-zа-яё0-9]+', text.lower())
        stop_words = {
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
            'такой','им','более','всегда','конечно','всю','между'
        }
        return [w for w in words if len(w) > 2 and w not in stop_words]

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
        Сохраняет факт в файл памяти с дедупликацией (векторная + Жаккард).

        :param confidence: доверие к факту, см. _CONFIDENCE_TAG_RE выше.
            confidence <= FACT_CONFIDENCE_DROP_THRESHOLD трактуется как
            РЕТРАКЦИЯ: вместо записи новой строки ищутся и удаляются похожие
            старые строки (это опровержение, а не новый факт) — если только
            они не являются anchor-строками (confidence >= FACT_CONFIDENCE_ANCHOR_THRESHOLD),
            которые сон-консолидация и обычное опровержение не имеют права трогать.
        Возвращает строку статуса.
        """
        try:
            validated = self._validate_path(path)
            full_path = os.path.join(self.base_dir, f"{validated}.md")

            with self._file_lock:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                new_fact = content.strip()
                is_retraction = confidence <= FACT_CONFIDENCE_DROP_THRESHOLD

                if is_retraction and not os.path.exists(full_path):
                    return "[MEMORY] Опровержение принято, но похожих фактов не найдено."

                if not is_retraction:
                    # Векторная проверка на дубли во всей базе (ретракции это не касается —
                    # мы ищем что удалить, а не проверяем "уже есть ли такое").
                    dup_check = self.vector_engine.search(new_fact, top_k=1)
                    if dup_check and dup_check[0]['score'] > VECTOR_DUPLICATE_THRESHOLD:
                        if dup_check[0]['id'] != validated:
                            return "[MEMORY] ACK. Факт уже существует в памяти под другим путём."

                lines_to_write = []
                new_words = set(w for w in re.findall(r'\w+', new_fact.lower()) if len(w) > 2)
                removed_count = 0

                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8") as f:
                        existing_lines = f.readlines()

                    for line in existing_lines:
                        clean_line = line.strip("- ").strip()
                        if not clean_line:
                            continue
                        line_without_ts = re.sub(r'\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', clean_line)
                        old_confidence, line_without_tag = _strip_confidence_tag(line_without_ts)
                        old_words = set(w for w in re.findall(r'\w+', line_without_tag.lower()) if len(w) > 2)
                        is_anchor = old_confidence >= FACT_CONFIDENCE_ANCHOR_THRESHOLD

                        if not new_words or not old_words:
                            lines_to_write.append(line)
                            continue
                        intersection = len(new_words.intersection(old_words))
                        union = len(new_words.union(old_words))
                        jaccard = intersection / union if union > 0 else 0

                        if jaccard >= JACCARD_DUPLICATE_THRESHOLD:
                            if is_anchor:
                                # Anchor-факты (подтверждённая истина) неприкосновенны:
                                # ни обычная перезапись, ни ретракция не могут их убрать.
                                lines_to_write.append(line)
                            else:
                                removed_count += 1
                                continue  # пропускаем старую строку (заменяем/удаляем)
                        else:
                            lines_to_write.append(line)

                if is_retraction:
                    # Ретракция ничего не добавляет — только удаляет похожее.
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

        # Векторный поиск
        vector_results = self.vector_engine.search(query, top_k=top_k)
        if vector_results:
            output = []
            for res in vector_results:
                content = res['text']
                if len(content) > 300:
                    content = content[:300] + "\n[...ОБРЕЗАНО]"
                output.append(f"Файл: {res['id']} (Сходство: {res['score']:.2f})\n{content}\n")
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
                         max_chars: int = AUTO_CONTEXT_MAX_CHARS) -> str:
        """
        Извлекает релевантные строки из памяти для автоматической инъекции в контекст.
        """
        vector_results = self.vector_engine.search(query, top_k=max_files)
        if not vector_results:
            return ""

        query_words = set(w for w in re.findall(r'[a-zа-яё0-9]+', query.lower()) if len(w) > 2)

        context_lines = []
        normalized_lines = set()
        total_len = 0

        for res in vector_results:
            lines = res['text'].split('\n')
            file_snippets = []

            for line in lines:
                snippet = re.sub(r'^[-*\s]+', '', line).strip()
                snippet = re.sub(r'^\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', snippet).strip()
                _, snippet = _strip_confidence_tag(snippet)
                snippet = snippet.strip()
                if not snippet:
                    continue
                snippet_words = set(w for w in re.findall(r'[a-zа-яё0-9]+', snippet.lower()) if len(w) > 2)
                if query_words.intersection(snippet_words):
                    file_snippets.append(snippet)
                if len(file_snippets) >= max_lines:
                    break

            # Фолбэк: если нет совпадений по словам, берём последнюю непустую строку
            if not file_snippets:
                for line in reversed(lines):
                    snippet = re.sub(r'^[-*\s]+', '', line).strip()
                    snippet = re.sub(r'^\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', snippet).strip()
                    _, snippet = _strip_confidence_tag(snippet)
                    snippet = snippet.strip()
                    if snippet:
                        file_snippets.append(snippet)
                        break

            for snippet in file_snippets:
                norm = re.sub(r'[^a-zа-яё0-9]', '', snippet.lower())
                if not norm or norm in normalized_lines:
                    continue
                normalized_lines.add(norm)
                to_add = f"- {snippet}."
                if total_len + len(to_add) > max_chars:
                    break
                context_lines.append(to_add)
                total_len += len(to_add)

            if total_len >= max_chars:
                break

        return "\n".join(context_lines) if context_lines else ""

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
   - О пользователе (вкусы, имя, привычки) -> [user/...] (пример: [user/dislikes] Не любит ботов).
   - О тебе (твои мысли, реакции, ошибки) -> [system/yui/yui_...] 
   - О ПК, железе, коде -> [system/pc/pc_...]
   - О животных, растениях, природе -> ОБЯЗАТЕЛЬНО СОЗДАВАЙ НОВУЮ ПАПКУ (пример: [biology/axolotls] Аксолотли это...).
   - Об остальных вещах (еда, фильмы) -> ОБЯЗАТЕЛЬНО СОЗДАВАЙ НОВУЮ ПАПКУ (пример: [food/sweets] Любит трубочки).
4. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО смешивать темы. Аксолотли НЕЛЬЗЯ класть в папку user или system. Для них всегда создается своя тема.
5. Если фактов нет, пиши только: NULL
6. ОПРОВЕРЖЕНИЕ: если из истории ясно, что пользователь ИСПРАВИЛ тебя, и что-то,
   во что ты раньше верила, ОКАЗАЛОСЬ НЕПРАВДОЙ — не пиши новый факт поверх старого,
   а ЯВНО ОПРОВЕРГНИ его меткой (c=-1) сразу после категории:
   [user/dislikes] (c=-1) Не любит собак.
   Это УДАЛИТ противоречащую запись из памяти вместо того, чтобы обе версии лежали
   рядом. Используй ТОЛЬКО когда действительно есть прямое опровержение, не для
   обычных новых фактов (у обычных фактов метки быть не должно).

История:
{hist_str}

Факты:"""