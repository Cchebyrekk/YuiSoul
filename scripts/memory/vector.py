# -*- coding: utf-8 -*-
"""
Векторный поиск на основе SentenceTransformer.
Использует модель intfloat/multilingual-e5-base на CPU.

Индекс — по отдельным фактам (строкам файлов памяти), а не по файлу целиком: один вектор
на весь файл размывался по мере роста файла, и посторонние запросы проходили порог чаще
нужных (замерено в evals/retrieval_eval.py). Поиск по-прежнему возвращает файлы — с
лучшим score среди их строк и списком подошедших строк.
"""
import os
import re
import json
import math
import datetime
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.config import (
    MEMORY_DIR,
    VECTOR_MODEL_NAME,
    VECTOR_DEVICE,
    VECTOR_SEARCH_THRESHOLD,
    RECENCY_BOOST_WEIGHT,
    RECENCY_HALFLIFE_DAYS,
)

INDEX_VERSION = 2
_FACT_PREFIX_RE = re.compile(r'^[-*\s]*(?:\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*)?(?:\(c=[+-]?[\d.]+\)\s*)?')


def fact_lines(text: str) -> list:
    """Текст фактов файла без маркеров списка, даты и метки доверия; заголовки и пустые строки пропускаются."""
    facts = []
    for line in text.split("\n"):
        if line.strip().startswith("#"):
            continue
        fact = _FACT_PREFIX_RE.sub("", line).strip()
        if fact:
            facts.append(fact)
    return facts


class VectorSearchEngine:
    def __init__(self, model_name: str = VECTOR_MODEL_NAME, device: str = VECTOR_DEVICE):
        print(f"[VECTOR] Загрузка модели эмбеддингов ({model_name}) на {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.index_dir = MEMORY_DIR
        self.vectors_file = os.path.join(self.index_dir, "vectors.npy")
        self.meta_file = os.path.join(self.index_dir, "metadata.json")
        self.threshold = VECTOR_SEARCH_THRESHOLD
        # Индекс пишут фоновые потоки (extract_and_save_facts, рефлексия), а читает поиск
        # в начале каждого хода. Без блокировки удаление/добавление сдвигает индексы
        # строк посреди перебора в search() -> IndexError или текст чужого документа.
        # Кодирование моделью (долгое) идёт вне блокировки.
        self._lock = threading.RLock()

        self.vectors = np.zeros((0, 0))
        self.lines = []   # [{"id": doc_id, "text": факт}], по порядку совпадают со строками self.vectors
        self.docs = {}    # doc_id -> {"text": содержимое файла, "updated_at": iso}
        loaded = False
        if os.path.exists(self.vectors_file) and os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict) and meta.get("version") == INDEX_VERSION:
                    self.vectors = np.load(self.vectors_file)
                    self.lines, self.docs = meta["lines"], meta["docs"]
                    loaded = len(self.lines) == len(self.vectors)
            except (OSError, ValueError, KeyError):
                loaded = False
        if not loaded and os.path.isdir(self.index_dir):
            # Индекс старого формата (вектор на файл) или повреждён — перестраиваем из .md
            self.rebuild_index()

    @property
    def metadata(self) -> list:
        """Документы (файлы) индекса: [{"id", "text", "updated_at"}] — для совместимости и отладки."""
        with self._lock:
            return [{"id": doc_id, **doc} for doc_id, doc in self.docs.items()]

    def _save_index(self):
        """Сохраняет векторы и метаданные на диск."""
        np.save(self.vectors_file, self.vectors)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump({"version": INDEX_VERSION, "docs": self.docs, "lines": self.lines}, f, ensure_ascii=False, indent=1)

    def _encode_passages(self, facts: list) -> np.ndarray:
        # Для E5: документы маркируются префиксом "passage: "
        if not facts:
            return np.zeros((0, 0))
        return np.asarray(self.model.encode([f"passage: {f}" for f in facts], normalize_embeddings=True))

    def _drop_doc_locked(self, doc_id: str):
        keep = [i for i, line in enumerate(self.lines) if line["id"] != doc_id]
        if len(keep) != len(self.lines):
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0))
            self.lines = [self.lines[i] for i in keep]
        self.docs.pop(doc_id, None)

    def remove_document(self, doc_id: str):
        """
        Удаляет документ из индекса (например, когда ретракция стёрла
        последнюю строку файла и он стал пустым — иначе в индексе остаётся
        осиротевшая запись, указывающая на текст, которого больше нет на диске).
        Безопасно вызывать для несуществующего doc_id — это no-op.
        """
        with self._lock:
            if doc_id not in self.docs:
                return
            self._drop_doc_locked(doc_id)
            self._save_index()

    def add_document(self, doc_id: str, text: str):
        """
        Добавляет или обновляет документ (файл памяти): его факты индексируются по одному.
        doc_id – относительный путь (без .md), text – содержимое файла.
        Каждое обновление документа обновляет его updated_at — это то, что
        двигает его вверх при ранжировании по свежести в search().
        """
        facts = fact_lines(text)
        vectors = self._encode_passages(facts)
        now_iso = datetime.datetime.now().isoformat()
        with self._lock:
            self._drop_doc_locked(doc_id)
            if facts:
                self.vectors = vectors if len(self.lines) == 0 else np.vstack([self.vectors, vectors])
                self.lines.extend({"id": doc_id, "text": fact} for fact in facts)
                self.docs[doc_id] = {"text": text, "updated_at": now_iso}
            self._save_index()

    def _recency_factor(self, updated_at: str) -> float:
        """
        Экспоненциальное затухание свежести: 1.0 для только что обновлённого
        документа, 0.5 через RECENCY_HALFLIFE_DAYS, и так далее.
        Отсутствие/некорректность updated_at (старые индексы без этого поля)
        трактуется как «нейтрально старое» — фактор 0, без буста и без штрафа.
        """
        if not updated_at:
            return 0.0
        try:
            ts = datetime.datetime.fromisoformat(updated_at)
        except ValueError:
            return 0.0
        age_days = max(0.0, (datetime.datetime.now() - ts).total_seconds() / 86400.0)
        if RECENCY_HALFLIFE_DAYS <= 0:
            return 0.0
        return math.pow(0.5, age_days / RECENCY_HALFLIFE_DAYS)

    def _line_scores(self, query: str):
        """
        (scores, lines, docs) — косинус запроса с каждым фактом и согласованный снимок индекса,
        взятый за один захват блокировки: иначе фоновая запись между снимками давала файл без текста.
        """
        with self._lock:
            if len(self.lines) == 0:
                return None, [], {}
        # Для E5: запросы маркируются префиксом "query: "
        query_vector = self.model.encode(f"query: {query}", normalize_embeddings=True)
        with self._lock:
            if len(self.lines) == 0:  # индекс могли очистить, пока кодировали запрос
                return None, [], {}
            return np.dot(self.vectors, query_vector), list(self.lines), dict(self.docs)

    def search_lines(self, query: str, top_k: int = 3, threshold: float = -1.0) -> list:
        """Самые похожие отдельные факты: [{"id", "line", "score"}] по убыванию score."""
        scores, lines, _ = self._line_scores(query)
        if scores is None:
            return []
        order = np.argsort(-scores)[:top_k]
        return [{"id": lines[i]["id"], "line": lines[i]["text"], "score": float(scores[i])}
                for i in order if scores[i] > threshold]

    def search(self, query: str, top_k: int = 3, threshold: float = None) -> list:
        """
        Ищет top_k наиболее подходящих документов (файлов).
        Возвращает [{"id", "text", "score", "lines"}]: score — лучший косинус среди фактов
        файла, lines — факты файла, прошедшие порог, по убыванию сходства.

        Релевантность (порог threshold, по умолчанию self.threshold) решается ИСКЛЮЧИТЕЛЬНО по
        чистому косинусу отдельного факта — свежесть не может протащить нерелевантный документ
        мимо фильтра. Среди прошедших фильтр порядок взвешивается свежестью файла (updated_at).
        """
        threshold = self.threshold if threshold is None else threshold
        scores, lines, docs = self._line_scores(query)
        if scores is None:
            return []
        by_doc = {}
        for idx in np.argsort(-scores):
            score = float(scores[idx])
            if score <= threshold:
                break
            by_doc.setdefault(lines[idx]["id"], []).append((score, lines[idx]["text"]))

        ranked = sorted(by_doc.items(), key=lambda kv: kv[1][0][0] + RECENCY_BOOST_WEIGHT * self._recency_factor(
            docs[kv[0]].get("updated_at")), reverse=True)
        return [{"id": doc_id, "text": docs[doc_id].get("text", ""), "score": hits[0][0],
                 "lines": [text for _, text in hits], "hits": hits} for doc_id, hits in ranked[:top_k]]

    def rebuild_index(self):
        """Перестраивает индекс из всех .md файлов в каталоге индекса."""
        all_docs = []
        for root, _, files in os.walk(self.index_dir):
            for file in files:
                if not file.endswith(".md"):
                    continue
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        rel_path = os.path.relpath(fpath, self.index_dir).replace("\\", "/")
                        if rel_path.endswith(".md"):
                            rel_path = rel_path[:-3]
                        # mtime файла — лучшее доступное приближение к "когда
                        # факт реально обновлялся", раз при полном ребилде
                        # своя история updated_at по документам теряется.
                        mtime_iso = datetime.datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat()
                        all_docs.append((rel_path, content, mtime_iso))
                except Exception:
                    continue

        lines, docs = [], {}
        for rel_path, content, mtime_iso in all_docs:
            facts = fact_lines(content)
            if facts:
                lines.extend({"id": rel_path, "text": fact} for fact in facts)
                docs[rel_path] = {"text": content, "updated_at": mtime_iso}
        vectors = self._encode_passages([line["text"] for line in lines])

        with self._lock:
            self.vectors, self.lines, self.docs = vectors, lines, docs
            if os.path.isdir(self.index_dir):
                self._save_index()
        print(f"[VECTOR] Индекс перестроен: {len(docs)} документов, {len(lines)} фактов.")
