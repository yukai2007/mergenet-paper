#!/usr/bin/env python3
"""Read-only audit of a collaborator packet; no weights or training imports.

Usage: python verify_packet_logs.py --packet-root /path/to/share
Only the JSON audit is written. Source references are relative to the packet.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

TOLERANCE = 0.000051  # Logs round epoch averages to four decimal places.
TRAIN = re.compile(r"Train:\s*(\d+)\s*\[\s*(\d+)/(\d+)")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
EMA = re.compile(
    rf"Test \(EMA\):\s*\[\s*(\d+)/(\d+)\].*?"
    rf"Loss:\s*(?:{NUMBER})\s*\(\s*({NUMBER})\).*?"
    rf"Acc@1:\s*(?:{NUMBER})\s*\(\s*({NUMBER})\).*?"
    rf"Acc@5:\s*(?:{NUMBER})\s*\(\s*({NUMBER})\)", re.I)
EMA_END = re.compile(r"Test \(EMA\):\s*\[\s*(\d+)/(\d+)\]")
METRICS = ("eval_loss", "eval_top1", "eval_top5")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching(values, row):
    return all(math.isfinite(x) and math.isfinite(float(row[k]))
               and abs(x - float(row[k])) <= TOLERANCE
               for k, x in zip(METRICS, values))


def parse_log(path, root, rows_by_epoch):
    relative = path.relative_to(root).as_posix()
    epoch = None
    segment = 0
    events = []
    restarts = []
    unparsed = []
    orphan = []
    ema_lines = 0
    train_lines = 0
    line_count = 0
    content_hash = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for line_count, raw in enumerate(handle, 1):
            content_hash.update(raw)
            line = raw.decode("utf-8", errors="replace")
            train = TRAIN.search(line)
            if train:
                current = int(train[1])
                train_lines += 1
                if epoch is not None and current < epoch:
                    segment += 1
                    restarts.append({"line": line_count, "previous_epoch": epoch,
                                     "new_epoch": current, "new_segment": segment})
                epoch = current
            if "Test (EMA):" not in line:
                continue
            ema_lines += 1
            end = EMA_END.search(line)
            if not end or end[1] != end[2]:
                continue  # Intermediate validation batches are not epoch averages.
            result = EMA.search(line)
            if result is None:
                unparsed.append(line_count)
                continue
            if epoch is None:
                orphan.append(line_count)
                continue
            values = tuple(float(v) for v in result.groups()[2:])
            row = rows_by_epoch.get(epoch)
            events.append({"epoch": epoch, "values": values, "file": relative,
                           "line": line_count, "segment": segment,
                           "matches_summary": matching(values, row) if row else None})
    segments = []
    for sid in range(segment + 1):
        members = [e for e in events if e["segment"] == sid]
        segments.append({"segment": sid, "parsed_final_ema_events": len(members),
                         "first_epoch": min((e["epoch"] for e in members), default=None),
                         "last_epoch": max((e["epoch"] for e in members), default=None),
                         "matching_summary_epochs": sorted({e["epoch"] for e in members
                                                            if e["matches_summary"] is True}),
                         "mismatching_summary_epochs": sorted({e["epoch"] for e in members
                                                               if e["matches_summary"] is False})})
    return events, {"file": relative, "compressed_sha256": sha256(path),
                    "uncompressed_sha256": content_hash.hexdigest(),
                    "line_count": line_count, "train_lines": train_lines,
                    "ema_lines": ema_lines, "parsed_final_ema_events": len(events),
                    "unparsed_final_ema_lines": unparsed,
                    "final_ema_lines_without_epoch_context": orphan,
                    "epoch_regressions": restarts, "segments": segments}


def audit_run(summary, root):
    args_file = summary.with_name("args.yaml")
    # Only a top-level integer is needed; avoid importing a training environment.
    epoch_arg = re.findall(r"^epochs:\s*(\d+)\s*$", args_file.read_text(), re.M)
    target = int(epoch_arg[0]) if len(epoch_arg) == 1 else None
    with summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    finite_issues = []
    blank_cells = collections.Counter()
    epochs = []
    for lineno, row in enumerate(rows, 2):
        epochs.append(int(row["epoch"]))
        for key, value in row.items():
            if value is None or value == "":
                blank_cells[key] += 1
                if key in METRICS or key == "epoch":
                    finite_issues.append({"line": lineno, "column": key, "issue": "missing_required"})
                continue
            try:
                valid = math.isfinite(float(value))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                finite_issues.append({"line": lineno, "column": key, "issue": "nonfinite_or_nonnumeric"})
    contiguous = epochs == list(range(len(rows)))
    row_map = {int(r["epoch"]): r for r in rows}
    events, logs = [], []
    for path in sorted(summary.parent.glob("*.gz")):
        if "launcher" not in path.name and "stdout" not in path.name:
            continue
        parsed, log = parse_log(path, root, row_map)
        events.extend(parsed)
        logs.append(log)

    # An entire earlier attempt may be separated by a proven epoch regression.
    # Exclusion requires: all its observed summary metrics disagree, and a later
    # segment in this very file matches every available summary epoch. Retain all
    # conflicting values and source locations, regardless of exclusion.
    excluded = set()
    exclusions = []
    for log in logs:
        segments = log["segments"]
        full = [s["segment"] for s in segments
                if s["matching_summary_epochs"] == sorted(row_map)
                and not s["mismatching_summary_epochs"] and rows]
        for item in segments:
            if (item["mismatching_summary_epochs"] and not item["matching_summary_epochs"]
                    and any(s > item["segment"] for s in full)):
                key = (log["file"], item["segment"])
                excluded.add(key)
                exclusions.append({"file": key[0], "segment": key[1],
                                   "reason": "earlier_attempt_before_epoch_regression; later_segment_matches_full_summary",
                                   "excluded_epochs": item["mismatching_summary_epochs"]})

    by_epoch = collections.defaultdict(list)
    for event in events:
        by_epoch[event["epoch"]].append(event)
    row_checks = []
    raw_ambiguous = []
    duplicate_count = 0
    conflicts = []
    for lineno, row in enumerate(rows, 2):
        epoch = int(row["epoch"])
        candidates = by_epoch[epoch]
        groups = {tuple(e["values"]) for e in candidates}
        duplicate_count += len(candidates) - len(groups)
        if len(groups) > 1:
            raw_ambiguous.append(epoch)
        active = [e for e in candidates if (e["file"], e["segment"]) not in excluded]
        active_groups = {tuple(e["values"]) for e in active}
        matches = [e for e in active if e["matches_summary"] is True]
        if len(active_groups) > 1:
            status = "AMBIGUOUS"
        elif matches:
            status = "MATCH"
        elif active:
            status = "MISMATCH"
        else:
            status = "MISSING"
        row_checks.append({"epoch": epoch, "summary_line": lineno,
                           "summary_metrics": {k: float(row[k]) for k in METRICS},
                           "status": status,
                           "matching_sources": [{k: e[k] for k in ("file", "line", "segment")}
                                                for e in matches]})
        for event in candidates:
            if event["matches_summary"] is False:
                conflicts.append({"epoch": epoch, "file": event["file"], "line": event["line"],
                                  "segment": event["segment"],
                                  "logged_metrics": dict(zip(METRICS, event["values"])),
                                  "excluded_earlier_attempt": (event["file"], event["segment"]) in excluded})
    counts = collections.Counter(r["status"] for r in row_checks)
    coverage = ("NO_PARSEABLE_EMA_ENDPOINTS" if not events else
                "FULL_MATCH_WITH_DOCUMENTED_EARLIER_ATTEMPT" if counts["MATCH"] == len(rows) and exclusions else
                "FULL_MATCH" if rows and counts["MATCH"] == len(rows) else "INCOMPLETE_OR_CONFLICTING")
    completion = ("TARGET_COMPLETE" if rows and target == len(rows) and contiguous else
                  "PARTIAL_SNAPSHOT" if rows and contiguous and target and len(rows) < target else
                  "EMPTY_OR_INVALID")
    best = max(rows, key=lambda r: float(r["eval_top1"])) if rows else None
    return {"run": summary.parent.name, "stage": summary.parent.parent.name,
            "summary_file": summary.relative_to(root).as_posix(), "summary_sha256": sha256(summary),
            "args_file": args_file.relative_to(root).as_posix(), "args_sha256": sha256(args_file),
            "target_epochs": target, "summary_rows": len(rows), "completion": completion,
            "epochs_contiguous_from_zero": contiguous, "all_present_summary_cells_finite": not finite_issues,
            "finite_or_parse_issues": finite_issues, "blank_cells_by_column": dict(blank_cells),
            "best_top1": float(best["eval_top1"]) if best else None,
            "best_epoch": int(best["epoch"]) if best else None,
            "final_top1": float(rows[-1]["eval_top1"]) if rows else None,
            "log_coverage": coverage, "matched_rows": counts["MATCH"],
            "missing_epochs": [r["epoch"] for r in row_checks if r["status"] == "MISSING"],
            "mismatch_epochs": [r["epoch"] for r in row_checks if r["status"] == "MISMATCH"],
            "ambiguous_epochs": [r["epoch"] for r in row_checks if r["status"] == "AMBIGUOUS"],
            "raw_conflicting_epochs_before_attempt_resolution": raw_ambiguous,
            "duplicate_epoch_metric_events_collapsed": duplicate_count,
            "excluded_earlier_attempts": exclusions, "conflicting_log_events": conflicts,
            "logs": logs, "row_checks": row_checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("log-audit.json"))
    parser.add_argument("--packet-label", default="share_20260919")
    parser.add_argument("--audit-date", default="2026-09-21")
    args = parser.parse_args()
    root = args.packet_root.resolve()
    runs = [audit_run(path, root) for path in sorted((root / "10_runs").glob("*/*/summary.csv"))]
    totals = {"runs": len(runs), "target_complete_runs": sum(r["completion"] == "TARGET_COMPLETE" for r in runs),
              "partial_snapshot_runs": sum(r["completion"] == "PARTIAL_SNAPSHOT" for r in runs),
              "summary_rows": sum(r["summary_rows"] for r in runs),
              "matched_rows_after_documented_attempt_resolution": sum(r["matched_rows"] for r in runs),
              "missing_log_epochs": sum(len(r["missing_epochs"]) for r in runs),
              "mismatch_epochs": sum(len(r["mismatch_epochs"]) for r in runs),
              "unresolved_ambiguous_epochs": sum(len(r["ambiguous_epochs"]) for r in runs),
              "raw_conflicting_epochs_before_attempt_resolution": sum(len(r["raw_conflicting_epochs_before_attempt_resolution"]) for r in runs),
              "duplicate_epoch_metric_events_collapsed": sum(r["duplicate_epoch_metric_events_collapsed"] for r in runs),
              "nonfinite_or_parse_issues": sum(len(r["finite_or_parse_issues"]) for r in runs),
              "gzip_logs": sum(len(r["logs"]) for r in runs),
              "unparsed_final_ema_lines": sum(len(l["unparsed_final_ema_lines"]) for r in runs for l in r["logs"]),
              "final_ema_lines_without_epoch_context": sum(len(l["final_ema_lines_without_epoch_context"]) for r in runs for l in r["logs"])}
    result = {"schema_version": 1, "packet": args.packet_label, "audit_date": args.audit_date,
              "verifier_sha256": sha256(Path(__file__)), "tolerance_absolute": TOLERANCE,
              "metric_order": list(METRICS),
              "method": "Match final Test (EMA) running averages to summary eval_loss/top1/top5 by the latest Train epoch; collapse equal epoch/metric records across launcher and stdout; retain source lines and distinct-attempt conflicts.",
              "scope_and_limits": [
                  "This is consistency verification of delivered files, not independent model reevaluation or proof of checkpoint identity.",
                  "Target completion describes available summary rows; partial snapshots do not establish that the original job failed.",
                  "No parseable EMA records are explicitly missing evidence, never a successful cross-check.",
                  "stdout and launcher copies are not independent experiments or duplicate epochs.",
                  "Earlier attempts are excluded only with a recorded epoch regression and a later segment matching the complete available summary; conflicting records remain in the audit.",
                  "Only standard EMA eval_loss/top1/top5 are log-matched; full-compression metrics and other CSV columns receive finite-value checks only.",
                  "No raw environment lines or absolute source-host paths are exported."],
              "totals": totals, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
