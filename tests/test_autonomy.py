# -*- coding: utf-8 -*-
import queue
import threading
import time

import scripts.agent.autonomy as autonomy_mod
from scripts.agent.autonomy import AutonomyManager, mark_activity, seconds_since_activity


def test_activity_clock():
    mark_activity()
    assert seconds_since_activity() < 1


def test_is_due_and_user_activity_reset(monkeypatch):
    manager = AutonomyManager(queue.Queue(), idle_threshold=60)
    monkeypatch.setattr(autonomy_mod, "seconds_since_activity", lambda: 61)
    assert manager.is_due()
    monkeypatch.setattr(autonomy_mod, "seconds_since_activity", lambda: 5)
    assert not manager.is_due()
    manager._unanswered = 3
    manager.on_user_activity()
    assert manager._unanswered == 0


def run_loop_briefly(manager, seconds):
    manager.start()
    time.sleep(seconds)
    manager.stop()


def test_loop_queues_inner_turn_and_backs_off(monkeypatch):
    monkeypatch.setattr(autonomy_mod, "AUTONOMY_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(autonomy_mod, "seconds_since_activity", lambda: 300)
    q = queue.Queue()
    manager = AutonomyManager(q, idle_threshold=0, min_interval=0.2)
    run_loop_briefly(manager, 0.5)  # 0.2 -> затем пауза 0.4 (x2): успевает только одна-две мысли
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert 1 <= len(items) <= 2
    source, prompt, meta = items[0]
    assert source == "autonomy" and "5 мин" in prompt and meta["idle_min"] == 5
    assert "speak_aloud" in prompt and "task_complete" in prompt


def test_loop_waits_while_yui_is_busy(monkeypatch):
    monkeypatch.setattr(autonomy_mod, "AUTONOMY_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(autonomy_mod, "seconds_since_activity", lambda: 300)
    q, working, speaking = queue.Queue(), threading.Event(), threading.Event()
    speaking.set()
    manager = AutonomyManager(q, agent_is_working=working, tts_active=speaking, idle_threshold=0, min_interval=0)
    run_loop_briefly(manager, 0.2)
    assert q.empty()


def test_loop_waits_for_pause_in_conversation(monkeypatch):
    monkeypatch.setattr(autonomy_mod, "AUTONOMY_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(autonomy_mod, "seconds_since_activity", lambda: 1)
    q = queue.Queue()
    run_loop_briefly(AutonomyManager(q, idle_threshold=180, min_interval=0), 0.2)
    assert q.empty()
