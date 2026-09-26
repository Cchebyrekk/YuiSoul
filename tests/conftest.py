# -*- coding: utf-8 -*-
"""
Общие фикстуры тестов. Тесты не трогают настоящие memory/ и sessions/, не грузят
модель эмбеддингов и не ходят в llama-server: любой непропатченный HTTP-запрос падает.
"""
import hashlib
import json
import re

import numpy as np
import pytest

import scripts.memory.vector as vector_mod
from scripts.utils.http import SESSION


class BagOfWordsModel:
    """Замена SentenceTransformer: вектор по словам, поэтому похожие тексты близки по косинусу."""
    DIM = 512

    def __init__(self, *args, **kwargs):
        pass

    def _one(self, text):
        text = re.sub(r'^(query|passage): ', '', text)
        v = np.zeros(self.DIM)
        # Только слова из букв: даты/метки "(c=0.30)" в строках фактов не должны влиять на сходство
        for w in re.findall(r'[a-zа-яё]{3,}', text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, text, normalize_embeddings=True):
        if isinstance(text, list):
            return np.array([self._one(t) for t in text])
        return self._one(text)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_mod, "SentenceTransformer", BagOfWordsModel)
    monkeypatch.setattr(vector_mod, "MEMORY_DIR", str(tmp_path / "memory"))

    def no_network(*args, **kwargs):
        raise AssertionError("тест попытался сделать настоящий HTTP-запрос")

    monkeypatch.setattr(SESSION, "post", no_network)
    monkeypatch.setattr(SESSION, "get", no_network)


@pytest.fixture
def memory_dir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def mm(memory_dir):
    from scripts.memory.manager import MemoryManager
    return MemoryManager(base_dir=str(memory_dir))


def write_facts(memory_dir, rel_path, lines):
    """Создаёт файл памяти memory/<rel_path>.md со строками фактов."""
    path = memory_dir / f"{rel_path}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


class FakeResponse:
    def __init__(self, content="", status_code=200, lines=None):
        self.status_code = status_code
        self._content = content
        self._lines = lines or []
        self.text = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}

    def iter_lines(self):
        # Без явных строк стрима — тот же content одним SSE-чанком (для запросов со stream=True)
        for line in self._lines or (sse({"content": self._content}) if self._content else []):
            yield line.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sse(*chunks):
    """Строки SSE-стрима llama-server из дельт: sse({"content": "Привет"}, {"tool_calls": [...]})."""
    lines = [f"data: {json.dumps({'choices': [{'delta': d}]}, ensure_ascii=False)}" for d in chunks]
    return lines + ["data: [DONE]"]


@pytest.fixture
def llm(monkeypatch):
    """Подменяет SESSION.post: отдаёт заранее заданные ответы по очереди и запоминает запросы."""
    class FakeLLM:
        def __init__(self):
            self.responses = []
            self.requests = []

        def post(self, url, json=None, **kwargs):
            self.requests.append(json)
            if not self.responses:
                raise AssertionError("LLM вызвана больше раз, чем задано ответов")
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    fake = FakeLLM()
    monkeypatch.setattr(SESSION, "post", fake.post)
    return fake
