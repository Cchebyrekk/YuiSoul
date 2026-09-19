# -*- coding: utf-8 -*-
"""
Парсинг потокового ответа от LLM.
Извлекает содержимое, мысли (reasoning), вызовы инструментов.
Поддерживает нативный формат tool_calls из API, а также XML-теги <tool_call> и <output>.
"""
import json
import re
from typing import Dict, List, Tuple, Optional, Any


class StreamParser:
    """
    Накопление и парсинг потоковых данных от LLM.
    Поддерживает:
    - content (текст ответа)
    - reasoning_content (скрытые размышления, если модель их отделяет)
    - tool_calls (нативный формат OpenAI)
    - XML-теги <tool_call>{...}</tool_call> (фолбэк)
    - Разделение по </thought> (всё до тега — мысли, после — ответ)
    """

    def __init__(self):
        self.content_buffer = ""
        self.reasoning_buffer = ""
        self.tool_calls: List[Dict[str, Any]] = []
        # Для потоковой сборки tool_calls
        self._tool_call_map: Dict[int, Dict[str, Any]] = {}

    def feed_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """
        Обрабатывает один chunk из стрима.
        chunk_data — это парсенный JSON из ответа сервера.
        """
        delta = chunk_data.get('choices', [{}])[0].get('delta', {})

        # 1. Reasoning (скрытые мысли)
        if 'reasoning_content' in delta and delta['reasoning_content']:
            self.reasoning_buffer += delta['reasoning_content']

        # 2. Обычный текст
        if 'content' in delta and delta['content']:
            self.content_buffer += delta['content']

        # 3. Tool calls (нативный формат OpenAI)
        if 'tool_calls' in delta:
            for tc_chunk in delta['tool_calls']:
                idx = tc_chunk.get("index", 0)
                # Инициализируем запись, если её нет
                if idx not in self._tool_call_map:
                    self._tool_call_map[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""}
                    }
                tc = self._tool_call_map[idx]
                if tc_chunk.get("id"):
                    tc["id"] = tc_chunk["id"]
                if tc_chunk.get("function", {}).get("name"):
                    tc["function"]["name"] += tc_chunk["function"]["name"]
                if tc_chunk.get("function", {}).get("arguments"):
                    tc["function"]["arguments"] += tc_chunk["function"]["arguments"]

    def finalize(self) -> Tuple[str, str, List[Dict[str, Any]], Optional[Tuple[str, float]]]:
        """
        Завершает сборку и возвращает:
        - final_reply (очищенный текст ответа, без мыслей и тегов)
        - reasoning (текст размышлений)
        - tool_calls (список вызовов в формате OpenAI)
        - self_reported_emotion: (emotion, intensity) из тега <emotion>любопытство, 0.7</emotion>,
          который модель сама вписывает в ответ (по образцу kuni: эмоция как часть
          естественного самоотчёта, а не только угадывание по ключевым словам постфактум).
          None, если тега не было или он не распарсился.
        """
        # 1. Собираем нативные tool_calls из карты
        if self._tool_call_map:
            for idx in sorted(self._tool_call_map.keys()):
                tc = self._tool_call_map[idx]
                # Проверяем, что функция полностью собрана
                if tc["function"]["name"] and tc["function"]["arguments"]:
                    self.tool_calls.append(tc)

        # 2. Разделение по </thought>
        full_content = self.content_buffer
        reasoning = self.reasoning_buffer.strip()
        final_reply = full_content.strip()

        if '</thought>' in final_reply:
            parts = final_reply.split('</thought>')
            # Всё до последнего закрывающего тега — мысли
            reasoning_parts = parts[:-1]
            if reasoning_parts:
                # Добавляем к уже накопленному reasoning
                reasoning += "\n" + "\n".join(reasoning_parts).strip()
            # Всё после последнего </thought> — финальный ответ
            final_reply = parts[-1].strip()
        else:
            # Если тега нет, весь текст считается ответом (но мы всё равно проверим на tool_calls)
            pass

        # 3. Извлечение XML-тегов <tool_call> (фолбэк)
        # Ищем все <tool_call>{...}</tool_call>
        tc_matches = re.findall(r'<tool_call>\s*({.*?})\s*</tool_call>', final_reply, re.DOTALL)
        for json_str in tc_matches:
            try:
                call_data = json.loads(json_str)
                # Проверяем, что это не дубликат уже найденного нативного вызова
                # (можно по имени и аргументам, но для простоты добавим, если нет совпадения)
                is_dup = False
                for existing in self.tool_calls:
                    if existing["function"]["name"] == call_data.get("name") and \
                       existing["function"]["arguments"] == json.dumps(call_data.get("arguments", {})):
                        is_dup = True
                        break
                if not is_dup:
                    self.tool_calls.append({
                        "id": f"xml_{len(self.tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": call_data.get("name"),
                            "arguments": json.dumps(call_data.get("arguments", {}))
                        }
                    })
                # Удаляем тег из финального ответа
                final_reply = final_reply.replace(f"<tool_call>{json_str}</tool_call>", "").strip()
            except json.JSONDecodeError:
                pass

        # 4. Удаляем огрызки <output> (если модель их использовала)
        final_reply = re.sub(r'</?output>', '', final_reply).strip()

        # 5. Убираем лишние пустые строки и пробелы
        final_reply = re.sub(r'\n{3,}', '\n\n', final_reply)

        # 6. Если есть текст внутри тега <output>...</output>, извлекаем его (на случай, если модель не использовала </thought>)
        output_match = re.search(r'<output>(.*?)</output>', final_reply, re.DOTALL)
        if output_match:
            final_reply = output_match.group(1).strip()

        # 7. Финальная очистка от преамбул (если модель их всё-таки выдала)
        preamble_patterns = [
            r'^\s*Here\'s a thinking process:.*?(?=\n|<output>|$)',
            r'^\s*Thinking Process:.*?(?=\n|<output>|$)',
            r'^\s*The user wants me to perform.*?(?=\n|<output>|$)'
        ]
        for pattern in preamble_patterns:
            final_reply = re.sub(pattern, '', final_reply, flags=re.DOTALL).strip()

        # 8. Самоотчёт об эмоции: <emotion>название[, интенсивность 0-1]</emotion>.
        # Извлекаем и вырезаем ДО того, как final_reply уйдёт в TTS/печать.
        self_reported_emotion = None
        emotion_match = re.search(
            r'<emotion>\s*([a-zA-Zа-яёА-ЯЁ_\-]+)\s*(?:,\s*([\d.]+))?\s*</emotion>',
            final_reply
        )
        if emotion_match:
            name = emotion_match.group(1).strip().lower()
            try:
                intensity = float(emotion_match.group(2)) if emotion_match.group(2) else 0.6
            except ValueError:
                intensity = 0.6
            intensity = max(0.0, min(1.0, intensity))
            self_reported_emotion = (name, intensity)
            final_reply = re.sub(r'<emotion>.*?</emotion>', '', final_reply, flags=re.DOTALL).strip()

        return final_reply, reasoning, self.tool_calls, self_reported_emotion

    def reset(self):
        """Сбрасывает состояние парсера для нового запроса."""
        self.content_buffer = ""
        self.reasoning_buffer = ""
        self.tool_calls = []
        self._tool_call_map = {}