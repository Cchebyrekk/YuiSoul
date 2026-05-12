import os

class SoulManager:
    def __init__(self, base_dir: str = "memory"):
        self.base_dir = base_dir

    def _read_dir_facts(self, sub_dir: str, max_chars: int = 500) -> str:
        """Читает все .md файлы в подпапке с жестким лимитом символов"""
        target_dir = os.path.join(self.base_dir, sub_dir)
        if not os.path.exists(target_dir):
            return ""
        
        facts = []
        total_len = 0
        
        for root, _, files in os.walk(target_dir):
            for file in files:
                if not file.endswith(".md"): continue
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        # Убираем дефисы списков для экономии токенов
                        clean_content = content.replace("- ", "").replace("* ", "")
                        facts.append(clean_content)
                        total_len += len(clean_content)
                        if total_len > max_chars:
                            break
                except Exception:
                    continue
            if total_len > max_chars:
                break # Останавливаем чтение следующих файлов, если лимит выбран
                
        return "\n".join(facts).strip()

    def generate_soul_patch(self) -> str:
        """Анализирует новую структуру памяти и генерирует динамическую заплатку"""
        
        # 1. Профиль пользователя: собираем всё из /memory/user/
        user_facts = self._read_dir_facts("user", max_chars=500)
        
        # 2. Самоанализ YUI: собираем ТОЛЬКО файлы system/yui_*
        yui_facts = []
        system_dir = os.path.join(self.base_dir, "system/yui")
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
                            if yui_len > 300: break
                    except Exception:
                        continue
        
        patch_parts = []
        
        if user_facts:
            patch_parts.append(f"<user_profile>\n{user_facts}\n</user_profile>")
            
        if yui_facts:
            patch_parts.append(f"<self_reflection>\n{' '.join(yui_facts)}\n</self_reflection>")
        
        if not patch_parts:
            return ""
            
        return (
            "<soul_dynamic_state>\n"
            "На основе данных из долговременной памяти скорректируй отношение к пользователю. "
            "Адаптируй тон (если он новичок — объясняй, если опытный — можешь не снисходить).\n\n"
            + "\n\n".join(patch_parts) +
            "\n</soul_dynamic_state>"
        )