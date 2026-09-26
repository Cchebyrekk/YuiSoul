# -*- coding: utf-8 -*-
import pytest

import scripts.agent.prompt as prompt_mod
from conftest import write_facts
from scripts.agent.emotion import EmotionBridge
from scripts.memory.soul import SoulManager
from scripts.tools.registry import (PC_CONTROL_TOOL_NAMES, SPEAK_ALOUD_TOOL, TOOLS, build_registry, inner_tools)


def names(tools):
    return [t["function"]["name"] for t in tools]


def test_every_tool_schema_has_a_handler(mm):
    registry = build_registry(mm)
    assert set(names(TOOLS)) == set(registry)
    assert names(TOOLS)[-2:] == ["stay_silent", "task_complete"]


def test_inner_tools():
    inner = names(inner_tools())
    assert inner[-2:] == ["speak_aloud", "task_complete"]
    assert not set(inner) & (PC_CONTROL_TOOL_NAMES | {"wait", "stay_silent"})
    assert PC_CONTROL_TOOL_NAMES <= set(names(inner_tools(allow_pc_control=True)))
    assert SPEAK_ALOUD_TOOL not in TOOLS                     # в обычном разговоре его нет


def test_registry_memory_and_control_handlers(mm, memory_dir):
    registry = build_registry(mm)
    assert registry["save_memory"](path="user/pets", content="Кот рыжий", confidence="0.4") == "[MEMORY] OK"
    assert "(c=+0.40) Кот рыжий" in (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")
    assert "Кот рыжий" in registry["search_memory"](query="FILE:user/pets")
    assert registry["stay_silent"](reason="устала") == "SILENCE:устала"
    assert registry["task_complete"]() == "TASK_COMPLETE:"
    assert registry["wait"](seconds=0)


@pytest.mark.parametrize("text, emotion", [
    ("", "neutral"),
    ("просто текст", "neutral"),
    ("Мне так грустно и печально", "sad"),
    ("Хм, интересно, надо подумать", "thinking"),
])
def test_extract_emotion(text, emotion):
    assert EmotionBridge.extract_emotion(text)[0] == emotion


def test_emotion_bridge_is_a_quiet_stub():
    bridge = EmotionBridge(enabled=False)
    bridge.send_emotion("happy", 0.8)
    bridge.close()


def test_soul_patch(memory_dir):
    soul = SoulManager(base_dir=str(memory_dir))
    assert soul.generate_soul_patch() == ""
    write_facts(memory_dir, "user/pets", ["- Кот Барсик"])
    write_facts(memory_dir, "system/yui/yui_character", ["- Язвительная"])
    write_facts(memory_dir, "system/yui/other", ["- Не черта характера"])
    write_facts(memory_dir, "reflections/reflection_notes", ["# Заголовок", "- Пользователь любит котов"])
    patch = soul.generate_soul_patch()
    assert "<user_profile>\nКот Барсик\n</user_profile>" in patch
    assert "Язвительная" in patch and "Пользователь любит котов" in patch
    assert "Не черта характера" not in patch and "Заголовок" not in patch


def test_dynamic_state_and_static_prompt(memory_dir, monkeypatch):
    monkeypatch.setattr(prompt_mod, "get_hardware_context", lambda: "GPU OK")
    write_facts(memory_dir, "user/pets", ["- Кот"])
    state = prompt_mod.get_dynamic_state(memory_base_dir=str(memory_dir), soul_patch="SOUL")
    assert "GPU OK" in state and "Файлов в /memory: 1" in state and "SOUL" in state
    assert prompt_mod.build_system_prompt() == prompt_mod.build_system_prompt()   # статичен ради KV-кэша
    assert "<inner_thought>" in prompt_mod.build_system_prompt()


def test_hardware_context_without_nvidia_smi(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(prompt_mod.subprocess, "run", missing)
    assert prompt_mod.get_hardware_context() == "GPU: Данные недоступны"
