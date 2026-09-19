# -*- coding: utf-8 -*-
"""
Векторный поиск на основе SentenceTransformer.
Использует модель intfloat/multilingual-e5-base на CPU.
"""
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.config import MEMORY_DIR, VECTOR_MODEL_NAME, VECTOR_DEVICE, VECTOR_SEARCH_THRESHOLD

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

    def add_document(self, doc_id: str, text: str):
        """
        Добавляет или обновляет документ в векторной базе.
        doc_id – относительный путь (без .md).
        text – содержимое файла.
        """
        # Для E5: документы маркируются префиксом "passage: "
        vector = self.model.encode(f"passage: {text}", normalize_embeddings=True)

        if len(self.vectors) == 0:
            self.vectors = np.array([vector])
            self.metadata = [{"id": doc_id, "text": text}]
        else:
            # Проверяем, существует ли уже такой doc_id
            existing_idx = next((i for i, m in enumerate(self.metadata) if m["id"] == doc_id), None)
            if existing_idx is not None:
                # Обновление
                self.vectors[existing_idx] = vector
                self.metadata[existing_idx]["text"] = text
            else:
                # Добавление
                self.vectors = np.vstack([self.vectors, vector])
                self.metadata.append({"id": doc_id, "text": text})
        self._save_index()

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Ищет top_k наиболее похожих документов.
        Возвращает список словарей с полями id, text, score.
        """
        if len(self.vectors) == 0:
            return []

        # Для E5: запросы маркируются префиксом "query: "
        query_vector = self.model.encode(f"query: {query}", normalize_embeddings=True)

        # Косинусное сходство (векторы нормализованы)
        scores = np.dot(self.vectors, query_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > self.threshold:
                results.append({
                    "id": self.metadata[idx]["id"],
                    "text": self.metadata[idx]["text"],
                    "score": float(score)
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
                        all_docs.append((rel_path, content))
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
        self.metadata = [{"id": doc[0], "text": doc[1]} for doc in all_docs]
        self._save_index()
        print(f"[VECTOR] Индекс перестроен: {len(self.metadata)} документов.")