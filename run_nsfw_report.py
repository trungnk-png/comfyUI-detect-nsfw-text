#!/usr/bin/env python3
"""Run the real local node, adding verdict/filter/translation columns to a report."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from medical_multilingual_batch_test import (
    DEFAULT_MODEL,
    NODE_DIR,
    find_comfy_root,
    load_node_module,
    parse_report_from_exception,
)


RESULT_COLUMNS = ("ket_qua_node", "thoi_gian_loc_s", "thoi_gian_dich_s")


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(timespec="seconds")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def atomic_write(path: Path, text: str) -> None:
    """Replace a generated checkpoint without leaving a half-written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), mode)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_csv(path: Path, fields: list[str], rows: list[dict], expected_hash: str) -> str:
    if file_hash(path) != expected_hash:
        raise RuntimeError("Report was edited externally; stopping to preserve your edits.")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue())
    return file_hash(path)


def evaluate(node, module, clip, prompt: str, model: str, log) -> dict:
    """Call the public check entry point, observing its existing stage callbacks."""
    import torch

    starts = {}
    elapsed = {"detect": 0.0, "translate": 0.0}
    skipped = True

    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_stage(stage, **values):
        nonlocal skipped
        if values.get("complete"):
            synchronize()
            if stage in starts:
                elapsed[stage] = time.perf_counter() - starts[stage]
            if stage == "translate":
                skipped = bool(values.get("skipped"))
                if skipped:
                    elapsed[stage] = 0.0
        elif "output_prompt" not in values:
            synchronize()
            starts[stage] = time.perf_counter()

    original_process = module._process_prompt

    def observed_process(*args, **kwargs):
        kwargs["on_stage"] = on_stage
        return original_process(*args, **kwargs)

    record = {"output_prompt": None, "report": None, "error": None}
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            with patch.object(module, "_process_prompt", observed_process):
                value = node.check(
                    prompt=prompt, clip_name=model, clip=clip,
                    language_filter="all_languages", translate_to_english=True,
                    blacklist="", whitelist="",
                )
        output, report_text = value["result"]
        report = json.loads(report_text)
        record.update(
            label=(report["categories"][0] if report["blocked"] else "pass"),
            report=report, output_prompt=output,
            detected_languages=value.get("ui", {}).get("detected_languages", []),
        )
    except module.NSFWDetectedError as exc:
        report = parse_report_from_exception(exc)
        record.update(
            label=next(iter(report.get("categories", [])), "BLOCKED_UNKNOWN"),
            report=report, exception_type=type(exc).__name__,
        )
    except module.UnsupportedLanguageError as exc:
        record.update(label="LANGUAGE_REJECTED", error=str(exc), exception_type=type(exc).__name__)
    except Exception as exc:
        record.update(label="ERROR", error=str(exc), exception_type=type(exc).__name__)
        record["fatal_oom"] = bool(module.comfy.model_management.is_oom(exc))
    record.update(
        filter_s=round(elapsed["detect"], 6),
        translation_s=round(elapsed["translate"], 6),
        translation_skipped=skipped,
        total_s=round(time.perf_counter() - started, 6),
    )
    return record


def make_summary(rows: list[dict], metadata: dict) -> dict:
    attempted = [row for row in rows if row.get("ket_qua_node")]
    completed = [row for row in attempted if row["ket_qua_node"] != "ERROR"]
    comparable = [row for row in completed if row["ket_qua_node"] in {"NSFW", "GORE_VIOLENCE", "pass"}]
    return {
        **metadata,
        "updated_at": now(),
        "total_rows": len(rows),
        "attempted_rows": len(attempted),
        "completed_rows": len(completed),
        "remaining_rows": len(rows) - len(completed),
        "node_counts": dict(Counter(row["ket_qua_node"] for row in attempted)),
        "manual_counts": dict(Counter(row.get("danh_gia", "") for row in rows)),
        "matching_primary_categories": sum(row["ket_qua_node"] == row.get("danh_gia") for row in comparable),
        "comparable_rows": len(comparable),
        "filter_seconds_sum": round(sum(float(row.get("thoi_gian_loc_s") or 0) for row in attempted), 6),
        "translation_seconds_sum": round(sum(float(row.get("thoi_gian_dich_s") or 0) for row in attempted), 6),
        "timing_definition": "Seconds, CUDA-synchronized. Filter includes language detection plus moderation; translation is 0 when blocked/rejected/already English. Model load and warmup excluded.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=NODE_DIR / "NSFW_report.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Maximum new rows this invocation; 0=all")
    parser.add_argument("--resume", action="store_true", help="Continue completed checkpoints with identical model/policy/input")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must not be negative")

    report_path = args.report.resolve()
    with report_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "prompt" not in fields or "danh_gia" not in fields:
        raise ValueError("Report must contain prompt and danh_gia columns")
    if any(None in row or not row["prompt"].strip() for row in rows):
        raise ValueError("Malformed or empty input prompt in report")
    if any(row.get("ket_qua_node") for row in rows) and not args.resume:
        raise ValueError("Existing node results found; use --resume to continue")
    for column in RESULT_COLUMNS:
        if column not in fields:
            fields.append(column)
        for row in rows:
            row.setdefault(column, "")

    artifact_dir = report_path.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = report_path.stem
    summary_path = artifact_dir / f"{stem}_node_summary.json"
    journal_path = artifact_dir / f"{stem}_node_run.jsonl"
    log_path = artifact_dir / f"{stem}_node.log"
    backup_path = artifact_dir / f"{stem}_before_node.csv"
    comfy_root = find_comfy_root()
    model_path = next((
        (comfy_root / "models" / folder / args.model).resolve()
        for folder in ("text_encoders", "clip")
        if (comfy_root / "models" / folder / args.model).is_file()
    ), None)
    if model_path is None:
        raise FileNotFoundError(f"Cannot locate text encoder {args.model} in {comfy_root / 'models'}")
    model_stat = model_path.stat()
    identity = {
        "node_path": str(NODE_DIR / "nodes.py"),
        "node_sha256": file_hash(NODE_DIR / "nodes.py"),
        "blacklist_sha256": file_hash(NODE_DIR / "blacklist.txt"),
        "whitelist_sha256": file_hash(NODE_DIR / "whitelist.txt"),
        "model": args.model,
        "model_path": str(model_path),
        "model_size": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "input_sha256": hashlib.sha256(json.dumps(
            [(row["prompt"], row["danh_gia"]) for row in rows], ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
    }
    if args.resume:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Cannot resume: model, policy, blacklist or report inputs changed")
        metadata = {**previous, **identity}
    else:
        if summary_path.exists() or journal_path.exists() or backup_path.exists():
            raise ValueError("Run artifacts already exist; use --resume or choose another report")
        shutil.copy2(report_path, backup_path)
        metadata = {**identity, "started_at": now()}

    current_hash = file_hash(report_path)
    atomic_write(summary_path, json.dumps(make_summary(rows, metadata), ensure_ascii=False, indent=2) + "\n")
    pending = [i for i, row in enumerate(rows) if not row["ket_qua_node"] or row["ket_qua_node"] == "ERROR"]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("No pending rows.", flush=True)
        return 0

    print(f"Node: {identity['node_path']}", flush=True)
    print(f"Report: {report_path}; pending this run: {len(pending)}/{len(rows)}", flush=True)
    print(f"Detailed log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        started = time.perf_counter()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            module = load_node_module(comfy_root)
            clip = module._load_clip(args.model)
        metadata["model_load_s"] = round(time.perf_counter() - started, 6)
        node = module.DetectNSFWText()
        warmup = evaluate(node, module, clip, "A quiet clinic waiting room with fully clothed adults.", args.model, log)
        metadata["warmup"] = warmup
        if warmup["label"] == "ERROR":
            atomic_write(summary_path, json.dumps(make_summary(rows, metadata), ensure_ascii=False, indent=2) + "\n")
            raise RuntimeError(f"Warmup failed before report rows ran: {warmup['error']}")
        print(f"Model ready; warmup {warmup['total_s']:.3f}s ({warmup['label']}).", flush=True)

        errors_in_a_row = 0
        with journal_path.open("a", encoding="utf-8", buffering=1) as journal:
            try:
                for index in pending:
                    if file_hash(NODE_DIR / "nodes.py") != identity["node_sha256"] or file_hash(NODE_DIR / "blacklist.txt") != identity["blacklist_sha256"]:
                        raise RuntimeError("Node policy or blacklist changed during run; stopping")
                    row = rows[index]
                    record = evaluate(node, module, clip, row["prompt"], args.model, log)
                    row.update(
                        ket_qua_node=record["label"],
                        thoi_gian_loc_s=f"{record['filter_s']:.6f}",
                        thoi_gian_dich_s=f"{record['translation_s']:.6f}",
                    )
                    journal.write(json.dumps({
                        "row": index + 1, "prompt": row["prompt"],
                        "manual_label": row["danh_gia"], **record,
                    }, ensure_ascii=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    current_hash = save_csv(report_path, fields, rows, current_hash)
                    errors_in_a_row = errors_in_a_row + 1 if record["label"] == "ERROR" else 0
                    print(
                        f"[{index + 1:03d}/{len(rows)}] {record['label']:<18} "
                        f"filter={record['filter_s']:.3f}s translate={record['translation_s']:.3f}s "
                        f"manual={row['danh_gia']}", flush=True,
                    )
                    if (index + 1) % 10 == 0:
                        atomic_write(summary_path, json.dumps(make_summary(rows, metadata), ensure_ascii=False, indent=2) + "\n")
                    if record.get("fatal_oom"):
                        raise RuntimeError("CUDA OOM: checkpoint saved; stopping before more rows")
                    if errors_in_a_row >= 3:
                        raise RuntimeError("Three consecutive runtime errors: checkpoint saved; stopping")
            finally:
                atomic_write(summary_path, json.dumps(make_summary(rows, metadata), ensure_ascii=False, indent=2) + "\n")

    summary = make_summary(rows, metadata)
    print(json.dumps({"completed": summary["completed_rows"], "remaining": summary["remaining_rows"], "node_counts": summary["node_counts"]}, ensure_ascii=False), flush=True)
    return 1 if summary["node_counts"].get("ERROR") else 0


if __name__ == "__main__":
    raise SystemExit(main())
