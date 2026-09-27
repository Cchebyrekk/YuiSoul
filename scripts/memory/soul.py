# -*- coding: utf-8 -*-
"""
Генерация динамической "души" (soul patch) на основе профиля пользователя
и саморефлексии YUI из папок memory/user/ и memory/system/yui/.
"""
import os
import datetime
import glob
import re
from scripts.config import MEMORY_DIR

# Слова, по которым видно, что факт — про владельца или про ваши с ним отношения: гостю такое не показываем
_OWNER_MARKERS = re.compile(r"пользовател|хозя|\b(он|его|ему|им|ним|нём|него)\b|переписк|telegram|телеграм", re.IGNORECASE)


def guest_safe_fact(fact: str) -> bool:
    """Можно ли показать этот факт о Юи гостю: в нём нет ничего о владельце."""
    return not _OWNER_MARKERS.search(fact)

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

    def _read_recent_diary(self, max_entries: int = 4, max_chars: int = 500) -> str:
        """Последние записи эмоционального дневника (memory/diary/ГГГГ-ММ-ДД.md), новые — в конце."""
        files = sorted(f for f in glob.glob(os.path.join(self.base_dir, "diary", "*.md"))
                       if os.path.basename(f) != "dreams.md")
        entries = []
        for fpath in reversed(files):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
            except OSError:
                continue
            day = os.path.basename(fpath)[:-3]
            for line in reversed(lines):
                entries.append(f"[{day}] " + re.sub(r'^-\s*(\[[^\]]*\]\s*)?(\(c=[^)]*\)\s*)?', '', line))
                if len(entries) >= max_entries:
                    break
            if len(entries) >= max_entries:
                break
        text = "\n".join(reversed(entries))
        return text[-max_chars:]

    def _last_dream(self, max_age_hours: float = 24) -> str:
        """Последний сон (memory/diary/dreams.md), если он приснился не раньше max_age_hours назад."""
        try:
            with open(os.path.join(self.base_dir, "diary", "dreams.md"), "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except OSError:
            return ""
        if not lines:
            return ""
        m = re.match(r'^-\s*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*(.*)$', lines[-1])
        if not m:
            return ""
        age = datetime.datetime.now() - datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
        return m.group(2) if age.total_seconds() <= max_age_hours * 3600 else ""

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

    def generate_guest_patch(self) -> str:
        """
        Душа для разговора с гостем из Telegram: только черты и предпочтения самой Юи — без профиля владельца,
        рефлексий о нём и строк, где он упоминается (см. guest_safe_fact).
        """
        lines = []
        system_dir = os.path.join(self.base_dir, "system", "yui")
        if os.path.exists(system_dir):
            for file in sorted(os.listdir(system_dir)):
                if not (file.startswith("yui_") and file.endswith(".md")):
                    continue
                try:
                    with open(os.path.join(system_dir, file), "r", encoding="utf-8") as f:
                        for line in f:
                            fact = re.sub(r'^-\s*(\[[^\]]*\]\s*)?(\(c=[^)]*\)\s*)?', '', line.strip())
                            if fact and guest_safe_fact(fact):
                                lines.append(fact)
                except Exception:
                    continue
        text = " ".join(lines)[:1000]
        return f"<self_reflection>\n{text}\n</self_reflection>" if text else ""

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
                            # Характер теперь растёт в основном через память, а не через промпт — лимит шире
                            if yui_len > 1000:
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

        # 4. Эмоциональный дневник (идея из kuni): что она недавно чувствовала — и последний сон, если был недавно
        feelings = self._read_recent_diary()
        if feelings:
            patch_parts.append(f"<recent_feelings>\n{feelings}\n</recent_feelings>")
        dream = self._last_dream()
        if dream:
            patch_parts.append(f"<last_dream>\n{dream}\n(Этот сон снился тебе недавно — можешь рассказать о нём, "
                               f"если захочется и будет к месту.)\n</last_dream>")

        if not patch_parts:
            return ""
        
        return (
            "<soul_dynamic_state>\n"
            "Это то, что ты накопила в памяти о себе и о пользователе, — часть твоего характера "
            "и отношения к нему, дополняющая базовое описание личности.\n\n"
            + "\n\n".join(patch_parts) +
            "\n</soul_dynamic_state>"
        )