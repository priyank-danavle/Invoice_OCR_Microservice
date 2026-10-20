"""
tests/benchmark_fuzzy.py
━━━━━━━━━━━━━━━━━━━━━━━━
Fuzzy matching accuracy benchmark.

Usage
─────
  # From the repo root (with the entity index already loaded):
  python -m tests.benchmark_fuzzy --csv tests/fuzzy_ground_truth_sample.csv

  # With a custom ground-truth file:
  python -m tests.benchmark_fuzzy --csv /path/to/my_ground_truth.csv

Ground-truth CSV format
───────────────────────
  raw,master,client_id,expected_id,expected_label,notes
  Reliance Inds.,supplier,1,42,Reliance Industries Ltd,abbreviation
  kgs,unit,1,3,kg,synonym

  • raw          – what OCR (or a user) extracted
  • master       – master_key (supplier / transport / color / unit …)
  • client_id    – integer client scope
  • expected_id  – the CORRECT entity PK (int)
  • expected_label – human-readable label for the correct entity
  • notes        – optional, printed in the failure report

Metrics reported
────────────────
  Overall accuracy  – % rows where matched_id == expected_id
  Per-master        – accuracy broken down by master_key
  Confidence dist.  – avg confidence on correct vs incorrect matches
  Failure report    – every row that was wrong, with the fuzzy winner shown
"""

from __future__ import annotations

import argparse
import csv
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

# ── Allow running as "python -m tests.benchmark_fuzzy" from project root ──────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.indexer import load as load_index
from app.ocr.matcher import EntityMatchError, match_entity

import asyncio


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    raw: str
    master: str
    client_id: int
    expected_id: int
    expected_label: str
    notes: str = ""


@dataclass
class Result:
    row: Row
    matched_id: Optional[int]
    matched_label: Optional[str]
    confidence: float
    correct: bool
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(path: str) -> List[Row]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, rec in enumerate(reader, start=2):  # row 1 = header
            try:
                rows.append(Row(
                    raw=rec["raw"].strip(),
                    master=rec["master"].strip(),
                    client_id=int(rec["client_id"]),
                    expected_id=int(rec["expected_id"]),
                    expected_label=rec["expected_label"].strip(),
                    notes=rec.get("notes", "").strip(),
                ))
            except (KeyError, ValueError) as exc:
                print(f"  [WARN] Skipping CSV row {i}: {exc}")
    return rows


def run_benchmark(rows: List[Row]) -> List[Result]:
    results = []
    for row in rows:
        try:
            match = match_entity(
                raw=row.raw,
                master_key=row.master,
                client_id=row.client_id,
                parent_id=None,
            )
            correct = (match.matched_id == row.expected_id)
            results.append(Result(
                row=row,
                matched_id=match.matched_id,
                matched_label=match.matched_label,
                confidence=match.confidence,
                correct=correct,
            ))
        except EntityMatchError as exc:
            results.append(Result(
                row=row,
                matched_id=None,
                matched_label=None,
                confidence=0.0,
                correct=False,
                error=str(exc),
            ))
    return results


def print_report(results: List[Result]) -> None:
    total = len(results)
    correct_count = sum(1 for r in results if r.correct)

    print("\n" + "═" * 65)
    print("  FUZZY MATCHING ACCURACY BENCHMARK")
    print("═" * 65)
    print(f"\n  Total test cases  : {total}")
    print(f"  Correct matches   : {correct_count}")
    print(f"  Incorrect matches : {total - correct_count}")
    print(f"\n  ✅ Overall Accuracy: {correct_count / total * 100:.1f}%")

    # ── Per-master breakdown ──────────────────────────────────────────────────
    by_master: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_master[r.row.master].append(r)

    print("\n" + "─" * 65)
    print("  PER-MASTER ACCURACY")
    print("─" * 65)
    print(f"  {'Master':<20} {'Cases':>6}  {'Correct':>8}  {'Accuracy':>10}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*8}  {'─'*10}")
    for master, mrs in sorted(by_master.items()):
        n = len(mrs)
        c = sum(1 for r in mrs if r.correct)
        bar = "█" * int(c / n * 20)
        print(f"  {master:<20} {n:>6}  {c:>8}  {c/n*100:>9.1f}%  {bar}")

    # ── Confidence distribution ───────────────────────────────────────────────
    correct_conf  = [r.confidence for r in results if r.correct]
    wrong_conf    = [r.confidence for r in results if not r.correct and not r.error]

    print("\n" + "─" * 65)
    print("  CONFIDENCE DISTRIBUTION")
    print("─" * 65)
    if correct_conf:
        avg_c = sum(correct_conf) / len(correct_conf)
        print(f"  Avg confidence (correct matches)   : {avg_c:.4f}")
    if wrong_conf:
        avg_w = sum(wrong_conf) / len(wrong_conf)
        print(f"  Avg confidence (incorrect matches) : {avg_w:.4f}")

    # Confidence bucket breakdown
    print("\n  Confidence buckets (all results):")
    buckets = {"HIGH ≥0.85": 0, "MEDIUM 0.60–0.84": 0, "LOW <0.60": 0}
    for r in results:
        if r.confidence >= 0.85:
            buckets["HIGH ≥0.85"] += 1
        elif r.confidence >= 0.60:
            buckets["MEDIUM 0.60–0.84"] += 1
        else:
            buckets["LOW <0.60"] += 1
    for label, cnt in buckets.items():
        print(f"    {label:<22}: {cnt:>3}  ({cnt/total*100:.0f}%)")

    # ── Failure report ────────────────────────────────────────────────────────
    failures = [r for r in results if not r.correct]
    if failures:
        print("\n" + "─" * 65)
        print(f"  FAILURE REPORT  ({len(failures)} failed)")
        print("─" * 65)
        for r in failures:
            print(f"\n  raw          : '{r.row.raw}'")
            print(f"  master       : {r.row.master}  (client_id={r.row.client_id})")
            print(f"  expected     : id={r.row.expected_id}  label='{r.row.expected_label}'")
            if r.error:
                print(f"  result       : ⚠ EntityMatchError — {r.error}")
            else:
                print(f"  got          : id={r.matched_id}  label='{r.matched_label}'  conf={r.confidence:.4f}")
            if r.row.notes:
                print(f"  notes        : {r.row.notes}")
    else:
        print("\n  🎉 No failures — perfect score!")

    print("\n" + "═" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────

async def _main(csv_path: str) -> None:
    print(f"  Loading entity index …")
    await load_index()
    print(f"  Loading ground truth: {csv_path}")
    rows = load_ground_truth(csv_path)
    if not rows:
        print("  No valid rows found in CSV. Exiting.")
        sys.exit(1)
    print(f"  Running {len(rows)} test cases …")
    results = run_benchmark(rows)
    print_report(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fuzzy matching accuracy against a ground-truth CSV."
    )
    parser.add_argument(
        "--csv",
        default="tests/fuzzy_ground_truth_sample.csv",
        help="Path to the ground-truth CSV file.",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.csv))


if __name__ == "__main__":
    main()
