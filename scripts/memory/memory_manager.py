import os
import re
import json
from typing import List, Dict, Any
import datetime

class MemoryManager:
    def __init__(self, base_dir: str = "memory"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _validate_path(self, path: str) -> str:
        """Проверяет лимит вложенности и чистит мусор"""
        # Нормализация слешей под Windows
        clean_path = path.replace("\\", "/").strip("/")
        
        # Убираем расширение, если модель по глупости его написала
        if clean_path.endswith(".md"): 
            clean_path = clean_path[:-3]
            
        # ОТСЕКАЕМ КОРНЕВУЮ ПАПККУ: если модель скопировала "memory/", отсекаем её
        base_name = os.path.basename(self.base_dir).lower()
        if clean_path.lower().startswith(f"{base_name}/"):
            clean_path = clean_path[len(base_name)+1:]
        elif clean_path.lower() == base_name:
            clean_path = ""
        
        parts = clean_path.split("/")
        
        # Лимит: 2 папки + 1 файл (макс 3 элемента)
        if len(parts) > 3: 
            raise ValueError(f"Лимит вложенности превышен. Запрошено уровней: {len(parts)}, макс: 2 (папка/папка/файл).")
            
        # Защита от пустого пути после чистки
        return clean_path if clean_path else "misc/default"

    def save_fact(self, path: str, content: str) -> str | dict:
        try:
            validated = self._validate_path(path)
            full_path = os.path.join(self.base_dir, f"{validated}.md")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            new_fact = content.strip()
            lines_to_write = []
            
            # Множество значимых слов нового факта
            new_words = set(w for w in re.findall(r'\w+', new_fact.lower()) if len(w) > 2)

            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    existing_lines = f.readlines()

                for line in existing_lines:
                    clean_line = line.strip("- ").strip()
                    if not clean_line: 
                        continue
                    
                    # Убираем таймстемп из старой строки перед расчетом схожести, чтобы дата не ломала Жаккара
                    line_without_ts = re.sub(r'\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', clean_line)
                    old_words = set(w for w in re.findall(r'\w+', line_without_ts.lower()) if len(w) > 2)
                    
                    if not new_words or not old_words:
                        lines_to_write.append(line)
                        continue
                        
                    # Стандартный Индекс Жаккара: пересечение / объединение
                    intersection = len(new_words.intersection(old_words))
                    union = len(new_words.union(old_words))
                    jaccard = intersection / union if union > 0 else 0
                    
                    if jaccard > 0.7: 
                        # Явный дубль (совпало больше 70% слов)
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        lines_to_write.append(f"- [{timestamp}] {new_fact}\n")
                    elif 0.4 <= jaccard <= 0.7: 
                        # СЕРАЯ ЗОНА: Возвращаем конфликт в агентный цикл
                        return {
                            "status": "conflict",
                            "path": full_path,
                            "old_fact": clean_line,
                            "new_fact": new_fact,
                            "similarity": f"{jaccard:.0%}"
                        }
                    else:
                        # Разные факты
                        lines_to_write.append(line)

            # Если конфликтов не было — дописываем/перезаписываем как обычно
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            lines_to_write.append(f"- [{timestamp}] {new_fact}\n")

            with open(full_path, "w", encoding="utf-8") as f:
                f.writelines(lines_to_write)

            return f"Факт в {full_path}: ДОБАВЛЕН/ОБНОВЛЕН"
        except ValueError as e:
            return f"Ошибка сохранения: {e}"
        
    def _get_tree(self) -> str:
        """Генерирует текстовое дерево директорий памяти"""
        tree_str = "/memory\n"
        for root, dirs, files in os.walk(self.base_dir):
            # Пропускаем скрытые папки и служебные файлы
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if f.endswith('.md')]
            
            level = root.replace(self.base_dir, '').count(os.sep)
            indent = '  ' * level
            tree_str += f'{indent}{os.path.basename(root)}/\n'
            subindent = '  ' * (level + 1)
            for file in files:
                tree_str += f'{subindent}{file}\n'
                
        return tree_str if len(tree_str) > 8 else "Память абсолютно пуста."

    def search_facts(self, query: str, top_k: int = 3) -> str:
        """Скрытый триггер: если query == '__tree__', возвращает структуру папок"""
        if query.strip() == "__tree__":
            return self._get_tree()
        
        # Скрытый триггер: если query == '__all__', возвращает ВСЕ файлы памяти
        if query.strip() == "__all__":
            all_content = []
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    if not file.endswith(".md"): continue
                    fpath = os.path.join(root, file)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if content:
                            rel_path = os.path.relpath(fpath, self.base_dir)
                            all_content.append(f"Файл: {rel_path}\n{content}")
                    except Exception:
                        continue
            return "\n\n".join(all_content) if all_content else "Память абсолютно пуста."
        
        # Скрытый триггер: если query начинается с "FILE:", читает файл по точному пути
        if query.strip().startswith("FILE:"):
            file_path = query.strip()[5:].strip() # Отрезаем "FILE:"
            if file_path.endswith(".md"): file_path = file_path[:-3] # Убираем .md
            
            full_path = os.path.join(self.base_dir, f"{file_path}.md")
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    return f"Содержимое {file_path}.md:\n{content}" if content else "Файл пуст."
                except Exception:
                    return "Ошибка чтения файла."
            return "Файл не найден."
            
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words: return "Пустой запрос."
        
        # ... (далее идет твой существующий код поиска без изменений) ...
        results = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if not file.endswith(".md"): continue
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    content_words = set(re.findall(r'\w+', content.lower()))
                    match_score = len(query_words.intersection(content_words))
                    
                    if match_score > 0:
                        rel_path = os.path.relpath(fpath, self.base_dir)
                        results.append({"path": rel_path, "score": match_score, "content": content})
                except Exception:
                    continue
                    
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]
        
        if not top_results:
            return "По памяти ничего не найдено."
            
        output = []
        MAX_RETURN_CHARS = 300 # Жесткий лимит выдачи по файлу
        for r in top_results:
            content = r['content']
            if len(content) > MAX_RETURN_CHARS:
                content = content[:MAX_RETURN_CHARS] + "\n[...ОБРЕЗАНО...]"
            output.append(f"Файл: {r['path']} (Релевантность: {r['score']})\nСодержимое:\n{content}\n")
            
        return "\n".join(output)

    def get_fact_extraction_prompt(self, history_to_compress: List[Dict[str, str]], tree_str: str = "") -> str:
        hist_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history_to_compress])
        
        tree_context = f"Текущая структура памяти:\n{tree_str}\n" if tree_str else ""
        
        return f"""Извлеки ТОЛЬКО важные долгосрочные факты.
{tree_context}
ЖЕСТКИЕ ПРАВИЛА ФОРМАТИРОВАНИЯ:
1. ЗАПРЕЩЕНО писать рассуждения, анализ, слова "Анализ", "Шаг". ПИШИ ТОЛЬКО ИТОГОВЫЕ ФАКТЫ.
2. Формат строго построчный: [категория/тема] Текст факта.
3. ПРАВИЛА КАТЕГОРИЙ (ВЫПОЛНЯТЬ СТРОГО):
   - О пользователе (вкусы, имя, привычки) -> [user/...] (пример: [user/dislikes] Не любит ботов).
   - О тебе (твои мысли, реакции, ошибки) -> [system/yui/yui_...] 
   - О ПК, железе, коде -> [system/pc/pc_...]
   - О животных, растениях, природе -> ОБЯЗАТЕЛЬНО СОЗДАВАЙ НОВУЮ ПАПКУ (пример: [biology/axolotls] Аксолотли это...).
   - Об остальных вещах (еда, фильмы) -> ОБЯЗАТЕЛЬНО СОЗДАВАЙ НОВУЮ ПАПКУ (пример: [food/sweets] Любит трубочки).
4. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО смешивать темы. Аксолотли НЕЛЬЗЯ класть в папку user или system. Для них всегда создается своя тема.
5. Если фактов нет, пиши только: NULL

История:
{hist_str}

Факты:"""