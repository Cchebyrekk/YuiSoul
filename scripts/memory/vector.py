# -*- coding: utf-8 -*-
"""
Векторный поиск на основе SentenceTransformer.
Использует модель intfloat/multilingual-e5-base на CPU.
"""
import os
import json
import math
import datetime
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

class VectorSearchEngine:
    def __init__(self, model_name: str = VECTOR_MODEL_NAME, device: str = VECTOR_DEVICE):
        print(f"[VECTOR] Загрузка модели эмбеддингов ({model_name}) на {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.index_dir = MEMORY_DIR
        self.vectors_file = os.path.join(self.index_dir, "vectors.npy")
        self.meta_file = os.path.join(self.index_dir, "metadata.json")
        self.threshold = VECTOR_SEARCH_THRESHOLD

        # Загружаем или создаём индекс
        if os.path.exists(self.vectors_file) and os.path.exists(self.meta_file):
            self.vectors = np.load(self.vectors_file)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.vectors = np.array([])
            self.metadata = []

    def _save_index(self):
        """Сохраняет векторы и метаданные на диск."""
        np.save(self.vectors_file, self.vectors)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def remove_document(self, doc_id: str):
        """
        Удаляет документ из индекса (например, когда ретракция стёрла
        последнюю строку файла и он стал пустым — иначе в индексе остаётся
        осиротевшая запись, указывающая на текст, которого больше нет на диске).
        Безопасно вызывать для несуществующего doc_id — это no-op.
        """
        if len(self.vectors) == 0:
            return
        existing_idx = next((i for i, m in enumerate(self.metadata) if m["id"] == doc_id), None)
        if existing_idx is None:
            return
        self.vectors = np.delete(self.vectors, existing_idx, axis=0)
        del self.metadata[existing_idx]
        self._save_index()

    def add_document(self, doc_id: str, text: str):
        """
        Добавляет или обновляет документ в векторной базе.
        doc_id – относительный путь (без .md).
        text – содержимое файла.
        Каждое обновление документа обновляет его updated_at — это то, что
        двигает его вверх при ранжировании по свежести в search().
        """
        # Для E5: документы маркируются префиксом "passage: "
        vector = self.model.encode(f"passage: {text}", normalize_embeddings=True)
        now_iso = datetime.datetime.now().isoformat()

        if len(self.vectors) == 0:
            self.vectors = np.array([vector])
            self.metadata = [{"id": doc_id, "text": text, "updated_at": now_iso}]
        else:
            # Проверяем, существует ли уже такой doc_id
            existing_idx = next((i for i, m in enumerate(self.metadata) if m["id"] == doc_id), None)
            if existing_idx is not None:
                # Обновление
                self.vectors[existing_idx] = vector
                self.metadata[existing_idx]["text"] = text
                self.metadata[existing_idx]["updated_at"] = now_iso
            else:
                # Добавление
                self.vectors = np.vstack([self.vectors, vector])
                self.metadata.append({"id": doc_id, "text": text, "updated_at": now_iso})
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

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Ищет top_k наиболее похожих документов.
        Возвращает список словарей с полями id, text, score.

        RAG 2.0 реранжирование: релевантность (порог self.threshold) решается
        ИСКЛЮЧИТЕЛЬНО по чистому косинусу — свежесть не может протащить
        нерелевантный документ мимо фильтра. Но среди уже прошедших фильтр
        порядок выдачи взвешивается свежестью (updated_at), чтобы недавно
        обновлённые/подтверждённые факты имели приоритет над устаревшими
        при равной релевантности.
        """
        if len(self.vectors) == 0:
            return []

        # Для E5: запросы маркируются префиксом "query: "
        query_vector = self.model.encode(f"query: {query}", normalize_embeddings=True)

        # Косинусное сходство (векторы нормализованы)
        cosine_scores = np.dot(self.vectors, query_vector)

        # Сначала фильтруем по релевантности (чистый косинус), потом ранжируем
        candidates = []
        for idx, score in enumerate(cosine_scores):
            if score > self.threshold:
                meta = self.metadata[idx]
                recency = self._recency_factor(meta.get("updated_at"))
                ranking_score = float(score) + RECENCY_BOOST_WEIGHT * recency
                candidates.append((idx, float(score), ranking_score))

        candidates.sort(key=lambda c: c[2], reverse=True)

        results = []
        for idx, cosine_score, _ranking_score in candidates[:top_k]:
            results.append({
                "id": self.metadata[idx]["id"],
                "text": self.metadata[idx]["text"],
                "score": cosine_score
            })
        return results

    def rebuild_index(self):
        """Перестраивает индекс из всех .md файлов в MEMORY_DIR."""
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

        if not all_docs:
            self.vectors = np.array([])
            self.metadata = []
            self._save_index()
            return

        # Строим векторы для всех документов
        texts = [f"passage: {doc[1]}" for doc in all_docs]
        vectors = self.model.encode(texts, normalize_embeddings=True)

        self.vectors = np.array(vectors)
        self.metadata = [{"id": doc[0], "text": doc[1], "updated_at": doc[2]} for doc in all_docs]
        self._save_index()
        print(f"[VECTOR] Индекс перестроен: {len(self.metadata)} документов.")