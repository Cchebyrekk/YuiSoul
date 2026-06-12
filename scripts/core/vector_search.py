import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorSearchEngine:
    def __init__(self, model_name="intfloat/multilingual-e5-base", device="cpu"):
        # ЗАГРУЗКА НА CPU. Безопасно для VRAM.
        print(f"[VECTOR] Загрузка модели эмбеддингов ({model_name}) на {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.index_dir = "memory"
        self.vectors_file = os.path.join(self.index_dir, "vectors.npy")
        self.meta_file = os.path.join(self.index_dir, "metadata.json")
        
        # Загружаем или создаем индекс
        if os.path.exists(self.vectors_file) and os.path.exists(self.meta_file):
            self.vectors = np.load(self.vectors_file)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.vectors = np.array([])
            self.metadata = []
            
    def _save_index(self):
        np.save(self.vectors_file, self.vectors)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_document(self, doc_id: str, text: str):
        """Добавляет документ в векторную базу"""
        # Специфика E5: для запросов добавляем "query: ", для документов "passage: "
        vector = self.model.encode(f"passage: {text}", normalize_embeddings=True)
        
        if len(self.vectors) == 0:
            self.vectors = np.array([vector])
        else:
            # Проверяем, есть ли уже этот doc_id (обновление)
            existing_idx = next((i for i, m in enumerate(self.metadata) if m["id"] == doc_id), None)
            if existing_idx is not None:
                self.vectors[existing_idx] = vector
                self.metadata[existing_idx]["text"] = text
                self._save_index()
                return
            
            self.vectors = np.vstack([self.vectors, vector])
            
        self.metadata.append({"id": doc_id, "text": text})
        self._save_index()

    def search(self, query: str, top_k: int = 3) -> list:
        """Ищет ближайшие по смыслу документы"""
        if len(self.vectors) == 0:
            return []
            
        query_vector = self.model.encode(f"query: {query}", normalize_embeddings=True)
        
        # Косинусное сходство (векторы уже нормализованы)
        scores = np.dot(self.vectors, query_vector)
        
        # Сортировка по убыванию
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            # ЖЕСТКИЙ ПОРОГ. 0.75 отсекает случайные совпадения (вроде "привет" = "любит трубочки")
            if score > 0.75: 
                results.append({
                    "id": self.metadata[idx]["id"],
                    "text": self.metadata[idx]["text"],
                    "score": float(score)
                })
        return results