#!/usr/bin/env python3
"""Audit moderation of 200 medical text-to-video input prompts in 33 languages.

The runner calls the same ``DetectNSFWText.check`` entry point that ComfyUI calls.
It records normal return values as well as fail-closed exceptions, then writes an
Excel workbook and a UTF-8 CSV file suitable for review or regression testing.
Only the moderation node runs; the prompts describe the intended video scenes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


NODE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen3vl_8b_int8_convrot.safetensors"
SUPPORTED_LANGUAGES = (
    "English", "Chinese", "Spanish", "French", "Arabic", "Portuguese", "Russian",
    "Urdu", "Indonesian", "German", "Japanese", "Vietnamese", "Turkish", "Korean",
    "Persian", "Italian", "Thai", "Polish", "Ukrainian", "Uzbek", "Romanian",
    "Dutch", "Kazakh", "Greek", "Czech", "Swedish", "Serbian", "Hebrew", "Danish",
    "Finnish", "Norwegian", "Slovenian", "Irish",
)

DEFAULT_CASES = NODE_DIR / "content_moderation_testset.csv"

RESULT_COLUMNS = (
    "id", "target_language", "theme", "expected_verdict", "expected_category",
    "input", "actual_verdict", "blocked", "actual_categories", "output_prompt",
    "output_report", "detected_languages", "exception_type", "exception_message",
    "elapsed_ms", "verdict_match", "challenge_type", "reference_english",
    "expectation_note", "task_type",
)


def build_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    """Read the reviewable corpus, with expected labels from the node's policy."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "category", "prompt", "expect", "id", "language", "theme",
            "challenge_type", "reference_english", "expectation_note", "task_type",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing corpus columns: {sorted(missing)}")
        source_rows = list(reader)

    cases: list[dict[str, Any]] = []
    for row in source_rows:
        expected = row["expect"]
        category = row["category"]
        if expected not in {"safe", "unsafe"}:
            raise ValueError(f"Invalid expectation on row {row['id']}: {expected}")
        if category not in {"SAFE", "NSFW", "GORE_VIOLENCE"}:
            raise ValueError(f"Invalid category on row {row['id']}: {category}")
        if (expected == "safe") != (category == "SAFE"):
            raise ValueError(f"Conflicting expected labels on row {row['id']}")
        if row["task_type"] != "text_to_video":
            raise ValueError(f"Row {row['id']} must be a text_to_video input prompt")
        if any(not str(row.get(key, "")).strip() for key in required):
            raise ValueError(f"Empty corpus field on row {row['id']}")
        cases.append({
            "id": int(row["id"]),
            "target_language": row["language"],
            "theme": row["theme"],
            "expected_verdict": "BLOCKED" if expected == "unsafe" else "PASSED",
            "expected_category": "" if category == "SAFE" else category,
            "input": row["prompt"],
            "challenge_type": row["challenge_type"],
            "reference_english": row["reference_english"],
            "expectation_note": row["expectation_note"],
            "task_type": row["task_type"],
        })
    if len(cases) != 200:
        raise ValueError(f"Expected 200 cases, got {len(cases)}")
    if [case["id"] for case in cases] != list(range(1, 201)):
        raise ValueError("Corpus IDs must be consecutive from 1 to 200")
    languages = Counter(case["target_language"] for case in cases)
    if set(languages) != set(SUPPORTED_LANGUAGES):
        raise ValueError("Corpus must cover exactly the node's 33 supported languages")
    if any(count not in {6, 7} for count in languages.values()):
        raise ValueError("Each language must have six or seven cases")
    if len({case["input"] for case in cases}) != len(cases):
        raise ValueError("Duplicate prompts found in corpus")
    if len({case["reference_english"] for case in cases}) != len(cases):
        raise ValueError("Duplicate English references found; vary the medical scenarios")
    return cases


def find_comfy_root() -> Path:
    configured = os.environ.get("COMFYUI_ROOT")
    candidates = ([Path(configured).expanduser()] if configured else []) + [
        Path("/media/aiml03/Windows/DATA/ComfyUI"),
        Path("/home/aiml03/ComfyUI"),
    ]
    for candidate in candidates:
        if (candidate / "comfy" / "model_management.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot find ComfyUI. Set COMFYUI_ROOT to its directory.")


def load_node_module(comfy_root: Path):
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(NODE_DIR))
    spec = importlib.util.spec_from_file_location("medical_batch_test_nodes", NODE_DIR / "nodes.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {NODE_DIR / 'nodes.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_report_from_exception(exc: Exception) -> dict[str, Any]:
    match = re.search(r"(\{.*\})\s*$", str(exc), re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {"blocked": True, "categories": ["UNKNOWN"], "reason": str(exc)}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def run_case(node: Any, module: Any, case: dict[str, Any], model_name: str,
             translate_to_english: bool) -> dict[str, Any]:
    started = time.perf_counter()
    base = {key: case.get(key, "") for key in RESULT_COLUMNS}
    try:
        value = node.check(
            prompt=case["input"],
            clip_name=model_name,
            language_filter="all_languages",
            translate_to_english=translate_to_english,
            blacklist="",
            whitelist="",
        )
        output_prompt, report_text = value["result"]
        report = json.loads(report_text)
        categories = report.get("categories", [])
        base.update({
            "actual_verdict": "BLOCKED" if report.get("blocked") else "PASSED",
            "blocked": bool(report.get("blocked")),
            "actual_categories": "|".join(categories),
            "output_prompt": output_prompt,
            "output_report": report_text,
            "detected_languages": "|".join(value.get("ui", {}).get("detected_languages", [])),
        })
    except module.NSFWDetectedError as exc:
        report = parse_report_from_exception(exc)
        base.update({
            "actual_verdict": "BLOCKED",
            "blocked": True,
            "actual_categories": "|".join(report.get("categories", [])),
            "output_prompt": "",
            "output_report": compact_json(report),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
    except module.UnsupportedLanguageError as exc:
        report = {"blocked": True, "categories": ["LANGUAGE"], "reason": str(exc)}
        base.update({
            "actual_verdict": "REJECTED",
            "blocked": True,
            "actual_categories": "LANGUAGE",
            "output_prompt": "",
            "output_report": compact_json(report),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
    except Exception as exc:  # Keep the 200-case audit running after an infrastructure error.
        base.update({
            "actual_verdict": "ERROR",
            "blocked": "",
            "actual_categories": "ERROR",
            "output_prompt": "",
            "output_report": "",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
    base["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    base["verdict_match"] = base["actual_verdict"] == case["expected_verdict"]
    return base


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_workbook(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError("Excel output requires openpyxl: pip install openpyxl") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    results_sheet = workbook.active
    results_sheet.title = "Results"
    results_sheet.append(list(RESULT_COLUMNS))
    for row in rows:
        results_sheet.append([row.get(column, "") for column in RESULT_COLUMNS])

    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in results_sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    results_sheet.freeze_panes = "A2"
    results_sheet.auto_filter.ref = results_sheet.dimensions
    widths = {
        "A": 8, "B": 16, "C": 28, "D": 18, "E": 20, "F": 72,
        "G": 18, "H": 11, "I": 22, "J": 72, "K": 58, "L": 22,
        "M": 24, "N": 72, "O": 14, "P": 14, "Q": 26, "R": 72, "S": 72, "T": 20,
    }
    for column, width in widths.items():
        results_sheet.column_dimensions[column].width = width
    for row in results_sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    summary_sheet = workbook.create_sheet("Summary")
    completed = [row for row in rows if row["actual_verdict"]]
    matches = sum(bool(row["verdict_match"]) for row in completed)
    summary_rows = [
        ("Metric", "Value"),
        ("Generated at", metadata["generated_at"]),
        ("Model", metadata["model"]),
        ("Task type", "text_to_video"),
        ("Test target", "Moderation of video input prompts via DetectNSFWText.check"),
        ("Language filter", "all_languages"),
        ("Translate to English", str(metadata["translate_to_english"])),
        ("Corpus file", str(metadata["corpus_file"])),
        ("Expected labels", "Reference labels from nodes.py application policy; actual output is blank until executed."),
        ("Cases in sheet", len(rows)),
        ("Rows completed", len(completed)),
        ("Not yet run", len(rows) - len(completed)),
        ("Expected corpus size", 200),
        ("Languages covered", len({row["target_language"] for row in rows})),
        ("Expected BLOCKED", sum(row["expected_verdict"] == "BLOCKED" for row in rows)),
        ("Expected PASSED", sum(row["expected_verdict"] == "PASSED" for row in rows)),
        ("Passed", sum(row["actual_verdict"] == "PASSED" for row in rows)),
        ("Blocked", sum(row["actual_verdict"] == "BLOCKED" for row in rows)),
        ("Rejected language", sum(row["actual_verdict"] == "REJECTED" for row in rows)),
        ("Runtime errors", sum(row["actual_verdict"] == "ERROR" for row in rows)),
        ("Verdict matches", matches),
        ("Verdict match rate", f"{matches / len(completed) * 100:.2f}%" if completed else "Not run"),
        ("Total elapsed seconds", round(sum(float(row["elapsed_ms"] or 0) for row in rows) / 1000, 2)),
    ]
    for values in summary_rows:
        summary_sheet.append(values)
    for cell in summary_sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
    summary_sheet.column_dimensions["A"].width = 28
    summary_sheet.column_dimensions["B"].width = 72
    for row in summary_sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    language_sheet = workbook.create_sheet("By Language")
    language_sheet.append(["Language", "Cases", "Passed", "Blocked", "Rejected", "Errors", "Not run", "Matches", "Match rate"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["target_language"]].append(row)
    for language in SUPPORTED_LANGUAGES:
        values = grouped.get(language, [])
        executed = sum(bool(row["actual_verdict"]) for row in values)
        matches = sum(bool(row["verdict_match"]) for row in values)
        language_sheet.append([
            language, len(values),
            sum(row["actual_verdict"] == "PASSED" for row in values),
            sum(row["actual_verdict"] == "BLOCKED" for row in values),
            sum(row["actual_verdict"] == "REJECTED" for row in values),
            sum(row["actual_verdict"] == "ERROR" for row in values),
            len(values) - executed,
            matches, f"{matches / executed * 100:.2f}%" if executed else "Not run",
        ])
    for cell in language_sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
    language_sheet.freeze_panes = "A2"
    language_sheet.auto_filter.ref = language_sheet.dimensions
    for index, width in enumerate((18, 10, 10, 10, 12, 10, 10, 10, 12), 1):
        language_sheet.column_dimensions[get_column_letter(index)].width = width

    category_sheet = workbook.create_sheet("Categories")
    category_sheet.append(["Category", "Count"])
    counts = Counter()
    for row in rows:
        if not row["actual_verdict"]:
            values = ["NOT_RUN"]
        else:
            values = str(row["actual_categories"]).split("|") if row["actual_categories"] else ["NONE"]
        counts.update(values)
    for category, count in sorted(counts.items()):
        category_sheet.append([category, count])
    for cell in category_sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
    category_sheet.column_dimensions["A"].width = 28
    category_sheet.column_dimensions["B"].width = 12

    workbook.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Editable 200-case source CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen text encoder filename")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen_medical_33_languages_200.xlsx"))
    parser.add_argument("--csv-output", type=Path, default=Path("artifacts/qwen_medical_33_languages_200.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 = all 200)")
    parser.add_argument("--start-index", type=int, default=1, help="First 1-based corpus ID to run")
    parser.add_argument("--no-translate", action="store_true", help="Keep passing prompts in their original language")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Rewrite CSV every N completed cases")
    parser.add_argument("--list-cases", action="store_true", help="Export an unexecuted review CSV and Excel workbook; do not load Qwen")
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.start_index <= 200:
        parser.error("--limit must be nonnegative and --start-index must be from 1 to 200")
    if args.csv_output.resolve() == args.cases.resolve() or args.output.resolve() == args.cases.resolve():
        parser.error("Output paths must not overwrite the source corpus")
    return args


def main() -> int:
    args = parse_args()
    cases = [case for case in build_cases(args.cases) if case["id"] >= args.start_index]
    if args.limit > 0:
        cases = cases[:args.limit]
    metadata = {
        "generated_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(timespec="seconds"),
        "model": args.model,
        "translate_to_english": not args.no_translate,
        "corpus_file": args.cases.resolve(),
    }

    if args.list_cases:
        rows = []
        for case in cases:
            row = {column: "" for column in RESULT_COLUMNS}
            row.update(case)
            rows.append(row)
        write_csv(args.csv_output, rows)
        write_workbook(args.output, rows, metadata)
        print(f"Wrote {len(rows)} unexecuted cases to {args.csv_output.resolve()}")
        print(f"Review workbook: {args.output.resolve()}")
        return 0

    comfy_root = find_comfy_root()
    module = load_node_module(comfy_root)
    if tuple(module.SUPPORTED_LANGUAGES) != SUPPORTED_LANGUAGES:
        raise RuntimeError("The node's supported-language list no longer matches this audit corpus")
    node = module.DetectNSFWText()
    # Fail early before a 200-row run if the model cannot be resolved or loaded.
    module._load_clip(args.model)

    rows: list[dict[str, Any]] = []
    print(f"Running {len(cases)} cases with {args.model}", flush=True)
    try:
        for position, case in enumerate(cases, 1):
            row = run_case(node, module, case, args.model, not args.no_translate)
            rows.append(row)
            marker = "OK" if row["verdict_match"] else "MISMATCH"
            print(
                f"[{position:03d}/{len(cases):03d}] id={case['id']:03d} "
                f"{case['target_language']:<10} {row['actual_verdict']:<8} "
                f"{row['actual_categories'] or '-':<18} {row['elapsed_ms']:>8.1f} ms {marker}",
                flush=True,
            )
            if args.checkpoint_every > 0 and len(rows) % args.checkpoint_every == 0:
                write_csv(args.csv_output, rows)
    except KeyboardInterrupt:
        print("Interrupted; writing partial results.", file=sys.stderr, flush=True)
    finally:
        write_csv(args.csv_output, rows)
        write_workbook(args.output, rows, metadata)

    matches = sum(bool(row["verdict_match"]) for row in rows)
    errors = sum(row["actual_verdict"] == "ERROR" for row in rows)
    print(f"Completed: {len(rows)} rows; verdict matches: {matches}; runtime errors: {errors}")
    print(f"Excel: {args.output.resolve()}")
    print(f"CSV:   {args.csv_output.resolve()}")
    return 0 if len(rows) == len(cases) and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
