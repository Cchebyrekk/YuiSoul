import os
import re
import json
from typing import List, Dict, Any
import datetime
from rank_bm25 import BM25Okapi
from core.vector_search import VectorSearchEngine

class MemoryManager:
    def __init__(self, base_dir: str = "memory"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.vector_engine = VectorSearchEngine(device="cpu")

    def _validate_path(self, path: str) -> str:
        """Проверяет лимит вложенности, чистит мусор и РЕЖЕТ длину пути (защита от спама LLM)"""
        # Нормализация слешей под Windows
        clean_path = path.replace("\\", "/").strip("/")
        
        # Убираем расширение, если модель по глупости его написала
        if clean_path.endswith(".md"): 
            clean_path = clean_path[:-3]
            
        # ОТСЕКАЕМ КОРНЕВУЮ ПАПКУ: если модель скопировала "memory/", отсекаем её
        base_name = os.path.basename(self.base_dir).lower()
        if clean_path.lower().startswith(f"{base_name}/"):
            clean_path = clean_path[len(base_name)+1:]
        elif clean_path.lower() == base_name:
            clean_path = ""
        
        parts = clean_path.split("/")
        
        # Лимит: 2 папки + 1 файл (макс 3 элемента)
        if len(parts) > 3: 
            raise ValueError(f"Лимит вложенности превышен. Запрошено уровней: {len(parts)}, макс: 2 (папка/папка/файл).")
            
        # --- ЗАЩИТА ОТ РЕКУРСИИ И ДЛИННЫХ ИМЕН ---
        sanitized_parts = []
        for part in parts:
            # Вырезаем недопустимые символы ОС Windows
            part = re.sub(r'[<>:"/\\|?*]', '', part)
            # Жесткая резка длины имени папки/файла (защита от confirmation_confirmation...)
            part = part[:25]
            # Windows крашится от точек и пробелов на концах имен
            part = part.strip(". ")
            if part:
                sanitized_parts.append(part)
                
        if not sanitized_parts:
            return "misc/default"
            
        final_path = "/".join(sanitized_parts)
        
        # Финальный предохранитель общей длины
        if len(final_path) > 80:
            final_path = final_path[:80]
            
        return final_path

    def _tokenize_russian(self, text: str) -> list:
        """Токенизация с удалением русских стоп-слов для повышения точности BM25"""
        words = re.findall(r'[a-zа-яё0-9]+', text.lower())
        
        # Базовый набор стоп-слов (местоимения, предлоги, союзы), которые сбивают BM25
        stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 
            'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 
            'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 
            'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 
            'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 
            'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 
            'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 
            'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 
            'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 
            'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 
            'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 
            'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 
            'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 
            'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 
            'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        }
        
        # Отсекаем слова < 3 символов И стоп-слова
        return [w for w in words if len(w) > 2 and w not in stop_words]

    def _bm25_search(self, query: str, top_k: int = 3) -> list:
        """Индексирует файлы в RAM на лету и ищет через BM25"""
        documents = []
        file_paths = []
        
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if not file.endswith(".md"): continue
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        documents.append(content)
                        file_paths.append(os.path.relpath(fpath, self.base_dir))
                except Exception:
                    continue
        
        if not documents: return []

        tokenized_corpus = [self._tokenize_russian(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = self._tokenize_russian(query)
        if not tokenized_query: return []
        
        scores = bm25.get_scores(tokenized_query)
        
        # Сортировка и фильтрация: отбрасываем всё с нулевым или отрицательным скором
        scored_results = sorted(zip(file_paths, documents, scores), key=lambda x: x[2], reverse=True)
        valid_results = [res for res in scored_results if res[2] > 0]
        
        return valid_results[:top_k]    

    def save_fact(self, path: str, content: str) -> str:
        try:
            validated = self._validate_path(path)
            full_path = os.path.join(self.base_dir, f"{validated}.md")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            new_fact = content.strip()
            
            # --- ВЕКТОРНАЯ ПРОВЕРКА НА ДУБЛИ МЕЖДУ ФАЙЛАМИ ---
            # Ищем, не сохраняли ли мы уже этот смысл в любой другой файл
            dup_check = self.vector_engine.search(new_fact, top_k=1)
            if dup_check and dup_check[0]['score'] > 0.90:
                # Если совпадение >90%, считаем что факт уже есть в системе.
                # Если путь совпадает, возможно мы обновляем, но если путь другой - это дубль.
                if dup_check[0]['id'] != validated:
                    return "[MEMORY] ACK. Факт уже существует в памяти под другим путем."
            # ------------------------------------------------

            lines_to_write = []
            new_words = set(w for w in re.findall(r'\w+', new_fact.lower()) if len(w) > 2)

            if os.path.exists(full_path):
                # ... дальше код без изменений (Жаккард внутри файла) ...
                with open(full_path, "r", encoding="utf-8") as f:
                    existing_lines = f.readlines()

                for line in existing_lines:
                    clean_line = line.strip("- ").strip()
                    if not clean_line: 
                        continue
                    
                    line_without_ts = re.sub(r'\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', clean_line)
                    old_words = set(w for w in re.findall(r'\w+', line_without_ts.lower()) if len(w) > 2)
                    
                    if not new_words or not old_words:
                        lines_to_write.append(line)
                        continue
                        
                    intersection = len(new_words.intersection(old_words))
                    union = len(new_words.union(old_words))
                    jaccard = intersection / union if union > 0 else 0
                    
                    # УБИТА СЕРАЯ ЗОНА. Если похоже (>= 0.4) — просто перезапишем старый факт новым.
                    if jaccard >= 0.4: 
                        continue
                    else:
                        lines_to_write.append(line)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            lines_to_write.append(f"- [{timestamp}] {new_fact}\n")

            with open(full_path, "w", encoding="utf-8") as f:
                f.writelines(lines_to_write)

            # --- ВЕКТОРИЗАЦИЯ СОХРАНЕННОГО ФАКТА ---
            # Читаем весь файл целиком и обновляем его вектор в базе
            with open(full_path, "r", encoding="utf-8") as f:
                full_content = f.read().strip()
            if full_content:
                self.vector_engine.add_document(doc_id=validated, text=full_content)

            return "[MEMORY] OK"
            
        except ValueError as e:
            return f"[MEMORY ERROR] {e}"
        
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

    def _extract_relevant_snippet(self, content: str, query_words: list, max_chars: int = 300) -> str:
        """Вырезает часть текста вокруг первого совпавшего слова"""
        content_lower = content.lower()
        min_idx = len(content)
        
        for word in query_words:
            idx = content_lower.find(word)
            if idx != -1 and idx < min_idx:
                min_idx = idx
                
        if min_idx == len(content):
            return content[:max_chars] # Фолбэк, если почему-то не нашли слово
            
        # Центрируем сниппет вокруг найденного слова
        start = max(0, min_idx - max_chars // 2)
        end = min(len(content), start + max_chars)
        
        snippet = content[start:end].strip()
        if start > 0: snippet = "[...]" + snippet
        if end < len(content): snippet += "[...ОБРЕЗАНО]"
        
        return snippet

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
            
        # --- ВЕКТОРНЫЙ ПОИСК (СМЫСЛОВОЙ) ---
        vector_results = self.vector_engine.search(query, top_k=top_k)
        
        if not vector_results:
            return "По памяти ничего не найдено."
            
        output = []
        MAX_RETURN_CHARS = 300
        for res in vector_results:
            content = res['text']
            if len(content) > MAX_RETURN_CHARS:
                content = content[:MAX_RETURN_CHARS] + "\n[...ОБРЕЗАНО]"
            output.append(f"Файл: {res['id']} (Смысловое совпадение: {res['score']:.2f})\nСодержимое:\n{content}\n")
            
        return "\n".join(output)

    def get_auto_context(self, query: str, max_files: int = 3, max_lines_per_file: int = 2, max_chars: int = 400) -> str:
        """Тихий поиск релевантных фактов для инъекции в системный промпт"""
        # Ищем top_k файлов по векторной базе
        vector_results = self.vector_engine.search(query, top_k=max_files)
        
        if not vector_results:
            return ""
            
        # Извлекаем значимые слова из запроса для точного поиска внутри файла
        query_words = set(w for w in re.findall(r'[a-zа-яё0-9]+', query.lower()) if len(w) > 2)
            
        context_lines = []
        normalized_lines = set() # Для жесткой дедупликации
        total_len = 0
        
        for res in vector_results:
            lines = res['text'].split('\n')
            file_snippets = []
            
            # Проходим по всем строкам файла
            for line in lines:
                snippet = re.sub(r'^[-*\s]+', '', line).strip()
                snippet = re.sub(r'^\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', snippet).strip()
                if not snippet: continue
                
                # Проверяем релевантность строки (пересечение слов запроса и строки)
                snippet_words = set(w for w in re.findall(r'[a-zа-яё0-9]+', snippet.lower()) if len(w) > 2)
                if query_words.intersection(snippet_words):
                    file_snippets.append(snippet)
                    
                # Жесткий лимит строк на один файл
                if len(file_snippets) >= max_lines_per_file:
                    break
            
            # Фолбэк: если векторный поиск дал файл, но слова вообще не совпали 
            # (чисто семантическая связь), берем первый попавшийся факт из файла
            if not file_snippets:
                for line in reversed(lines): # <--- ИДЕМ С КОНЦА ФАЙЛА
                    snippet = re.sub(r'^[-*\s]+', '', line).strip()
                    snippet = re.sub(r'^\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', snippet).strip()
                    if snippet:
                        file_snippets.append(snippet)
                        break # Найдя первую не пустую строку с конца, останавливаемся
            
            # Дедупликация и добавление в итоговый контекст
            for snippet in file_snippets:
                # Агрессивная нормализация для дедупликации
                norm_snippet = re.sub(r'[^a-zа-яё0-9]', '', snippet.lower())
                if not norm_snippet or norm_snippet in normalized_lines: continue
                normalized_lines.add(norm_snippet)
                
                snippet_to_add = f"- {snippet}."
                
                if total_len + len(snippet_to_add) > max_chars:
                    break
                    
                context_lines.append(snippet_to_add)
                total_len += len(snippet_to_add)
                
            if total_len >= max_chars:
                break
                
        return "\n".join(context_lines) if context_lines else ""
    
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
