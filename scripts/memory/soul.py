# -*- coding: utf-8 -*-
"""
Генерация динамической "души" (soul patch) на основе профиля пользователя
и саморефлексии YUI из папок memory/user/ и memory/system/yui/.
"""
import os
import glob
from scripts.config import MEMORY_DIR

class SoulManager:
    def __init__(self, base_dir: str = MEMORY_DIR):
        self.base_dir = base_dir

    def _read_dir_facts(self, sub_dir: str, max_chars: int = 500) -> str:
        """Читает все .md файлы в подпапке с жестким лимитом символов."""
        target_dir = os.path.join(self.base_dir, sub_dir)
        if not os.path.exists(target_dir):
            return ""
        
        facts = []
        total_len = 0
        
        for root, _, files in os.walk(target_dir):
            for file in files:
                if not file.endswith(".md"):
                    continue
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        # Убираем маркеры списков для экономии токенов
                        clean_content = content.replace("- ", "").replace("* ", "")
                        facts.append(clean_content)
                        total_len += len(clean_content)
                        if total_len > max_chars:
                            break
                except Exception:
                    continue
            if total_len > max_chars:
                break
        return "\n".join(facts).strip()

    def _read_recent_reflections(self, max_files: int = 2, max_chars: int = 400) -> str:
        """
        Читает N последних заметок ReflectionManager из /memory/reflections/.
        Имена файлов содержат сортируемую по времени метку
        (reflection_YYYY-MM-DD_HH-MM.md), поэтому сортировки по имени достаточно.
        Это и есть точка, где фоновая консолидация памяти (RAG 2.0) реально
        влияет на поведение — иначе рефлексии просто лежат мёртвым грузом в файлах.
        """
        reflections_dir = os.path.join(self.base_dir, "reflections")
        files = sorted(glob.glob(os.path.join(reflections_dir, "reflection_*.md")), reverse=True)

        notes = []
        total_len = 0
        for fpath in files[:max_files]:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                # Убираем markdown-заголовок "# Рефлексия от ..." — дата не нужна модели
                content = "\n".join(
                    line for line in content.split("\n") if not line.startswith("#")
                ).strip().replace("- ", "")
                if content:
                    notes.append(content)
                    total_len += len(content)
                    if total_len > max_chars:
                        break
            except Exception:
                continue
        return "\n".join(notes).strip()

    def generate_soul_patch(self) -> str:
        """
        Анализирует структуру памяти и генерирует динамическую заплатку:
        - user_profile из /memory/user/
        - self_reflection из /memory/system/yui/ + последние заметки ReflectionManager
        """
        # 1. Профиль пользователя
        user_facts = self._read_dir_facts("user", max_chars=500)

        # 2. Саморефлексия YUI (только файлы system/yui/yui_*.md)
        yui_facts = []
        system_dir = os.path.join(self.base_dir, "system", "yui")
        if os.path.exists(system_dir):
            yui_len = 0
            for file in os.listdir(system_dir):
                if file.startswith("yui_") and file.endswith(".md"):
                    fpath = os.path.join(system_dir, file)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read().strip().replace("- ", "")
                        if content:
                            yui_facts.append(content)
                            yui_len += len(content)
                            if yui_len > 300:
                                break
                    except Exception:
                        continue

        # 3. Последние выводы фоновой рефлексии (RAG 2.0 консолидация)
        recent_reflections = self._read_recent_reflections()

        patch_parts = []
        if user_facts:
            patch_parts.append(f"<user_profile>\n{user_facts}\n</user_profile>")
        if yui_facts or recent_reflections:
            reflection_block = " ".join(yui_facts)
            if recent_reflections:
                reflection_block = f"{reflection_block}\n{recent_reflections}".strip()
            patch_parts.append(f"<self_reflection>\n{reflection_block}\n</self_reflection>")

        if not patch_parts:
            return ""
        
        return (
            "<soul_dynamic_state>\n"
            "На основе данных из долговременной памяти скорректируй отношение к пользователю. "
            "Адаптируй тон (если он новичок — объясняй, если опытный — можешь не снисходить).\n\n"
            + "\n\n".join(patch_parts) +
            "\n</soul_dynamic_state>"
        )