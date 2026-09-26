# -*- coding: utf-8 -*-
"""Определение языка фразы по алфавиту — для выбора голоса TTS и языка озвучки эмодзи."""
import re

# Английской речью считаются слова с хотя бы двумя строчными латинскими буквами подряд:
# аббревиатуры ("Z.A.T.O.", "OK", "GPU") в русской фразе её язык не меняют.
_ENGLISH_WORD_RE = re.compile(r'[A-Za-z]*[a-z]{2,}[A-Za-z]*')
_CYRILLIC_RE = re.compile(r'[а-яА-ЯёЁ]')


def text_language(text: str, default: str = "ru") -> str:
    """
    "en" или "ru" по тому, каких букв во фразе больше. Без слов (эмодзи, цифры,
    одни аббревиатуры) — default: так эмодзи в середине ответа озвучивается на языке
    соседних фраз. Русский Silero латиницу внутри русской фразы пропускает, а целиком
    английский текст не читает — поэтому решение по фразе, а не по отдельным словам.
    """
    latin = sum(len(w) for w in _ENGLISH_WORD_RE.findall(text))
    cyrillic = len(_CYRILLIC_RE.findall(text))
    if not latin and not cyrillic:
        return default
    return "en" if latin > cyrillic else "ru"
