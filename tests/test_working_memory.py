# -*- coding: utf-8 -*-
import datetime
import queue

from scripts.memory.working import ReminderWatcher, WorkingMemory, working_memory_handler

NOW = datetime.datetime(2026, 9, 27, 20, 0)


def wm(tmp_path):
    return WorkingMemory(str(tmp_path / "working_memory.json"))


def test_add_shows_in_context_and_expires(tmp_path):
    w = wm(tmp_path)
    assert "id=1" in w.add("Обещала посмотреть его проект", days=2, now=NOW)
    ctx = w.context(now=NOW)
    assert ctx.startswith("<things_to_remember>") and "Обещала посмотреть его проект" in ctx
    assert w.context(now=NOW + datetime.timedelta(days=2)) != ""          # срок — включительно
    assert w.context(now=NOW + datetime.timedelta(days=3)) == ""          # потом исчезает само


def test_done_removes(tmp_path):
    w = wm(tmp_path)
    w.add("купить корм Барсику", now=NOW)
    assert "убрана" in w.done(1, now=NOW) and w.context(now=NOW) == ""
    assert "Нет записи" in w.done(5, now=NOW)


def test_reminder_fires_once(tmp_path):
    w = wm(tmp_path)
    assert "напомню 2026-09-28 10:00" in w.add("созвон с Леной", remind_at="2026-09-28 10:00", now=NOW)
    assert "(напомнить 2026-09-28 10:00)" in w.context(now=NOW)
    assert w.due(now=NOW) == []
    later = datetime.datetime(2026, 9, 28, 10, 0, 30)
    (item,) = w.due(now=later)
    assert item["text"] == "созвон с Леной"
    assert w.due(now=later) == []                                          # второй раз не срабатывает
    assert "напомнить" not in w.context(now=later)


def test_unfired_reminder_outlives_expiry(tmp_path):
    w = wm(tmp_path)
    w.add("через неделю продлить подписку", days=1, remind_at="2026-10-04 12:00", now=NOW)
    assert "продлить подписку" in w.context(now=datetime.datetime(2026, 10, 3, 9, 0))


def test_bad_times_are_rejected(tmp_path):
    w = wm(tmp_path)
    assert "не поняла" in w.add("x", remind_at="завтра утром", now=NOW)
    assert "уже прошло" in w.add("x", remind_at="2026-09-27 19:00", now=NOW)


def test_handler(tmp_path):
    w = wm(tmp_path)
    assert "Записала" in working_memory_handler(w, "add", text="позвонить маме")
    assert "позвонить маме" in working_memory_handler(w, "list")
    assert "укажи id" in working_memory_handler(w, "done")
    assert "убрана" in working_memory_handler(w, "done", item_id=1)
    assert working_memory_handler(w, "list") == "Рабочая память пуста."


def test_watcher_puts_reminder_turn(tmp_path):
    w = wm(tmp_path)
    w.add("выключить духовку", remind_at=(datetime.datetime.now() + datetime.timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M"))
    q = queue.Queue()
    watcher = ReminderWatcher(q, w, where=lambda: "Последний раз он писал из Telegram 5 мин. назад.")
    w.due = lambda now=None: [{"id": 1, "text": "выключить духовку", "remind_at": "2026-09-27 20:01"}]
    watcher.check()
    source, prompt, meta = q.get_nowait()
    assert source == "reminder" and meta["item_id"] == 1
    assert "выключить духовку" in prompt and "из Telegram" in prompt and "id=1" in prompt


def test_dynamic_state_includes_working_memory(tmp_path):
    import scripts.agent.prompt as prompt_mod
    WorkingMemory(str(tmp_path / "working_memory.json")).add("ждёт ответ про шлем")
    assert "ждёт ответ про шлем" in prompt_mod.get_dynamic_state(memory_base_dir=str(tmp_path))
