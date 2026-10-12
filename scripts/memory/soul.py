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


# Сколько символов «души» уходит в каждое сообщение — настоящие лимиты, свежее важнее старого
SOUL_USER_CHARS = 900      # что она знает о пользователе
SOUL_SELF_CHARS = 1400     # её характер и вкусы (system/yui/yui_*.md)
SOUL_NOTES_CHARS = 700     # последние заметки рефлексии
SOUL_DREAM_CHARS = 500     # последний сон
SOUL_LINE_MAX = 240        # длиннее — заметка-простыня, берём начало
_STAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")
_TOOL_MARKUP_RE = re.compile(r"</?(?:function|parameter|tool_call)\b")
# «- [дата] (c=+0.90) [дата] (дата) текст» -> «текст»
_PREFIX_RE = re.compile(r"^[-*]?\s*(?:\[[^\]]*\]\s*|\(c=[^)]*\)\s*|\(\d{4}-\d{2}-\d{2}[^)]*\)\s*)*")


def guest_safe_fact(fact: str) -> bool:
    """Можно ли показать этот факт о Юи гостю: в нём нет ничего о владельце."""
    return not _OWNER_MARKERS.search(fact)

class SoulManager:
    def __init__(self, base_dir: str = MEMORY_DIR):
        self.base_dir = base_dir

    def _newest_lines(self, pattern: str, max_chars: int) -> str:
        """
        Самые свежие строки фактов из файлов memory/<pattern> — не больше max_chars символов, по порядку времени,
        без дат и меток доверия. Раньше лимит только прекращал читать СЛЕДУЮЩИЕ файлы, а первый брался целиком:
        за три дня работы reflection_notes.md и yui_character.md выросли, «душа» стала 29 тыс. символов на
        каждое сообщение (половина всего контекста), и разговор из истории вытеснялся — Юи «тупела» (замерено).
        """
        dated = []
        for fpath in glob.glob(os.path.join(self.base_dir, *pattern.split("/")), recursive=True):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                fallback = datetime.datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M")
            except OSError:
                continue
            when = None  # дата последней датированной строки файла: строка без даты — продолжение той же заметки
            for line in lines:
                stamp = _STAMP_RE.search(line)
                when = stamp.group(1) if stamp else when
                text = _PREFIX_RE.sub("", line).strip()
                # Обрывки вызова инструмента, сохранённые как «факты» (замерено: <function=save_memory>,
                # <parameter=path>, голый путь reflections/reflection_notes), — не её мысли, в душу не идут
                if not text or _TOOL_MARKUP_RE.search(text) or ("/" in text and " " not in text):
                    continue
                dated.append((when or fallback, len(dated), text))
        picked, total = [], 0
        for _, _, text in sorted(dated, reverse=True):  # от свежих к старым, пока есть место
            if len(text) > SOUL_LINE_MAX:
                text = text[:SOUL_LINE_MAX].rstrip() + "…"
            if total + len(text) > max_chars:
                break
            picked.append(text)
            total += len(text) + 1
        return "\n".join(reversed(picked))

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

    def generate_soul_patch(self, include_notes: bool = True) -> str:
        """
        Анализирует структуру памяти и генерирует динамическую заплатку:
        - user_profile из /memory/user/
        - self_reflection из /memory/system/yui/ + последние заметки ReflectionManager
        """
        # 1. Профиль пользователя (memory/user/, с подпапками) — самое свежее в пределах лимита
        user_facts = self._newest_lines("user/**/*.md", SOUL_USER_CHARS)

        # 2. Саморефлексия YUI (только файлы system/yui/yui_*.md)
        yui_facts = self._newest_lines("system/yui/yui_*.md", SOUL_SELF_CHARS)

        # 3. Последние выводы фоновой рефлексии (RAG 2.0 консолидация)
        # Самой рефлексии её же заметки не показываем: она читала их и дописывала похожее — замкнутый круг
        recent_reflections = (self._newest_lines("reflections/reflection_*.md", SOUL_NOTES_CHARS)
                              if include_notes else "")

        patch_parts = []
        if user_facts:
            patch_parts.append(f"<user_profile>\n{user_facts}\n</user_profile>")
        if yui_facts or recent_reflections:
            reflection_block = f"{yui_facts}\n{recent_reflections}".strip()
            patch_parts.append(f"<self_reflection>\n{reflection_block}\n</self_reflection>")

        # 4. Эмоциональный дневник (идея из kuni): что она недавно чувствовала — и последний сон, если был недавно
        feelings = self._read_recent_diary()
        if feelings:
            patch_parts.append(f"<recent_feelings>\n{feelings}\n</recent_feelings>")
        dream = self._last_dream()
        if len(dream) > SOUL_DREAM_CHARS:
            dream = dream[:SOUL_DREAM_CHARS].rstrip() + "…"
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