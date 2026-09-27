# -*- coding: utf-8 -*-
"""
Проверка памяти без LLM: python -m evals.retrieval_eval [--show]

На настоящей модели эмбеддингов (e5) и временной копии памяти из evals/retrieval.yaml считает:
  recall     — доля запросов, для которых нужный факт попал в автоконтекст (его Юи видит каждый ход)
  noise      — доля «посторонних» запросов, на которые в автоконтекст всё равно что-то подмешалось
  dedup      — доля пар, где дубль/не дубль распознан верно (новый факт в том же файле и в другом)
Использует только публичные методы MemoryManager — годится для сравнения до/после правок.
"""
import argparse
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

TMP = tempfile.mkdtemp(prefix="yui_retrieval_")
import scripts.config as config  # noqa: E402
config.MEMORY_DIR = os.path.join(TMP, "memory")

import scripts.memory.vector as vector_mod  # noqa: E402

_models = {}


def _shared_embeddings(name, device=None):
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(name, device=device)
    return _models[name]


vector_mod.SentenceTransformer = _shared_embeddings

from scripts.memory.manager import MemoryManager  # noqa: E402


def fresh_memory(name):
    mem_dir = os.path.join(TMP, name)
    os.makedirs(mem_dir, exist_ok=True)
    vector_mod.MEMORY_DIR = mem_dir
    return MemoryManager(base_dir=mem_dir), mem_dir


def all_lines(mem_dir):
    lines = []
    for root, _, files in os.walk(mem_dir):
        for f in files:
            if f.endswith(".md"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    lines += [l for l in fh.read().splitlines() if l.strip()]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="печатать автоконтекст для каждого запроса")
    args = parser.parse_args()
    with open(os.path.join(ROOT, "evals", "retrieval.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mm, _ = fresh_memory("main")
    for path, facts in data["memory"].items():
        for fact in facts:
            mm.save_fact(path, fact)

    hits = 0
    print("--- запросы (нужный факт в автоконтексте?) ---")
    for item in data["queries"]:
        ctx = mm.get_auto_context(item["q"])
        ok = item["expect"].lower() in ctx.lower()
        hits += ok
        print(f"{'ok  ' if ok else 'MISS'} {item['q']:45} -> {ctx.replace(chr(10), ' | ')[:110] if (args.show or not ok) else ''}")

    noisy = 0
    print("--- посторонние запросы (автоконтекст должен быть пуст) ---")
    for q in data["negatives"]:
        ctx = mm.get_auto_context(q)
        noisy += bool(ctx)
        print(f"{'ok  ' if not ctx else 'NOISE'} {q:45} -> {ctx.replace(chr(10), ' | ')[:110]}")

    dedup_ok, dedup_total = 0, 0
    print("--- дедупликация (тот же файл / другой файл) ---")
    for i, pair in enumerate(data["dedup"]):
        verdicts = []
        for where in ("same", "other"):
            mm_d, mem_dir = fresh_memory(f"dedup_{i}_{where}")
            mm_d.save_fact("user/a", pair["old"])
            mm_d.save_fact("user/a" if where == "same" else "user/b", pair["new"])
            n = len(all_lines(mem_dir))
            detected_dup = n == 1
            ok = detected_dup == pair["dup"]
            dedup_ok += ok
            dedup_total += 1
            verdicts.append(f"{where}:{'ok' if ok else 'FAIL'}({n} стр.)")
        print(f"{'дубль  ' if pair['dup'] else 'разные '} {' '.join(verdicts):32} {pair['old'][:38]} || {pair['new'][:40]}")

    nq, nn = len(data["queries"]), len(data["negatives"])
    print(f"\nrecall {hits}/{nq} ({100 * hits // nq}%) | noise {noisy}/{nn} ({100 * noisy // nn}%) | "
          f"dedup {dedup_ok}/{dedup_total} ({100 * dedup_ok // dedup_total}%)")


if __name__ == "__main__":
    main()
