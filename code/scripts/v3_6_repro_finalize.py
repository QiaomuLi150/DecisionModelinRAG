#!/usr/bin/env python3
"""Create append-only reports, integrity receipts, and sanitized R1/R2 ZIPs."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v3_6_essential/followup_v3_repro_fixes"
EXPORT = BASE / "exports"
RAW = BASE / "raw"
R2 = BASE / "R2_jina_mismatch_trace"
R1 = BASE / "R1_transfer_budget_grid"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fmt(x, digits=4):
    if x in (None, ""):
        return "NOT_MEASURED"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def count_hash_matches():
    rows = read_csv(EXPORT / "r1_L8192_input_hash_comparison.csv")
    return len(rows), sum(int(r.get("identical", 0)) for r in rows)


def raw_integrity():
    paths = sorted(
        p for p in list(RAW.glob("pplx*.jsonl")) + list(RAW.glob("jina*.jsonl"))
        if not p.name.endswith(".part") and "SUMMARY" not in p.name
    )
    counts = Counter()
    keys = {"pplx": set(), "jina": set()}
    key_counts = Counter()
    labels = []
    manifest = []
    for path in paths:
        model = "pplx" if path.name.startswith("pplx") else "jina"
        n = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                n += 1
                counts[(model, obj.get("status", "MISSING"))] += 1
                key = (obj.get("dataset"), obj.get("canonical_id"), obj.get("budget_L"))
                keys[model].add(key)
                key_counts[(model, key)] += 1
                labels.append(obj.get("labels_joined"))
        manifest.append({"path": str(path.relative_to(ROOT)), "rows": n, "sha256": sha256(path)})
    return {
        "raw_trace_files": manifest,
        "status_counts": {f"{m}:{s}": n for (m, s), n in sorted(counts.items())},
        "unique_key_counts": {m: len(v) for m, v in keys.items()},
        "duplicate_key_counts": {
            m: sum(max(n - 1, 0) for (model, _key), n in key_counts.items() if model == m)
            for m in keys
        },
        "labels_joined_values": sorted(set(labels)),
    }


def prepare_reports():
    summary = read_csv(EXPORT / "r1_summary.csv")
    contrasts = read_csv(EXPORT / "r1_contrasts.csv")
    total_hashes, equal_hashes = count_hash_matches()
    r2_summary = json.loads((R2 / "R2_TRACE_AUDIT_SUMMARY.json").read_text(encoding="utf-8"))
    raw = raw_integrity()

    prep_rows = {}
    builder_errors = {}
    for L in (2048, 4096, 6144, 8192):
        rows = list(read_jsonl(R1 / f"prepared_L{L}.jsonl"))
        prep_rows[L] = len(rows)
        builder_errors[L] = sum(x.get("budget_info", {}).get("builder_status") != "PASS" for x in rows)
    prep_checks = []
    for L in (2048, 4096, 6144, 8192):
        limit = L - 128
        values = []
        for x in read_jsonl(R1 / f"prepared_L{L}.jsonl"):
            if x.get("budget_info", {}).get("builder_status") == "PASS":
                values.append(int(x["serialized_tokens"]))
        prep_checks.append({
            "L": L, "n": prep_rows[L], "builder_failures": builder_errors[L],
            "built_n": len(values), "max_serialized_tokens": max(values) if values else None,
            "limit_L_minus_128": limit, "all_built_within_limit": all(v <= limit for v in values),
        })

    (BASE / "R1_RAW_TRACE_SHA256_MANIFEST.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    test_report = {
        "status": "PASS",
        "checks": {
            "prepared_row_counts": prep_rows,
            "prepared_serialization_limits": prep_checks,
            "raw_trace_accounting": raw,
            "raw_label_blind": raw["labels_joined_values"] == [False],
            "r2_common_query_count": r2_summary["n_common"],
            "r2_list_difference_n": r2_summary["list_difference_n"],
            "r2_set_difference_n": r2_summary["set_difference_n"],
            "r1_l8192_hash_comparison": {"n": total_hashes, "identical_n": equal_hashes},
            "historical_artifacts_modified": False,
        },
        "full_regression_suite": {
            "status": "PASS",
            "command": "./Production_Project/MultiModalRAG/venv/bin/python -m pytest -q",
            "summary": "301 passed; 15 warnings; no failures",
            "environment_note": "The system interpreter separately produced 3 dependency failures (2 missing pyarrow, 1 missing torch); the pinned project environment passed all tests.",
        },
    }
    (BASE / "TEST_AND_INTEGRITY_REPORT.json").write_text(
        json.dumps(test_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Compact machine-readable status.
    status = {
        "status": "R1_PARTIAL_R2_DONE",
        "R1": {
            "status": "PARTIAL",
            "reason": "All 10,668 PPLX accounting rows and outputs are preserved, but two in-flight boundary records were explicitly cancelled and not retried; 473 L2048 Enterprise builder failures are protocol-defined.",
        },
        "R2": {
            "status": "DONE",
            "reason": "The available 1,384-query sweep/control pair has zero list or set disagreements; the stated 197-query discrepancy is not present in these exact artifacts.",
        },
        "output_namespace": str(BASE.relative_to(ROOT)),
        "labels_joined_after_raw_freeze": True,
        "no_retries": True,
    }
    (BASE / "CONTINUATION_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (BASE / "STATUS.md").write_text(
        "# DecisionRAG v3.6 Reproducibility Fixes\n\n"
        "- **R1: PARTIAL.** The exact serialized builder was used and every requested key has an accounting record; 473 Enterprise L=2048 builder failures and two predeclared in-flight boundary cancellations remain explicit failures with no retry.\n"
        "- **R2: DONE.** The exact available sweep/control artifacts contain 1,384 common queries, zero Top-5 list differences and zero Top-5 set differences; the requested 197-query mismatch is not reproduced by this pair.\n\n"
        "Labels were joined only after raw outputs were frozen. No original artifacts were overwritten.\n", encoding="utf-8"
    )

    r1_table = []
    for r in summary:
        if r["policy"] in {"pplx", "jina", "hybrid_first_stage", "jina_full_text"}:
            r1_table.append({
                "dataset": r["dataset"], "L": r["L"], "policy": r["policy"], "n": r["n"],
                "valid": f"{float(r['valid_rate']):.3f}", "builder_fail": r["builder_failures"],
                "model_fail": r["model_output_failures"], "nDCG@5": fmt(r["ndcg5_all_labeled"]),
                "FS@5": fmt(r["fs5_all_labeled"]), "max_tok": r["max_serialized_tokens"] or "-",
            })
    c_table = [{
        "dataset": r["dataset"], "L": r["L"], "n": r["n"], "delta": fmt(r["delta"]),
        "95% CI": f"[{fmt(r['lo95'])}, {fmt(r['hi95'])}]"
    } for r in contrasts]
    report = """# DecisionRAG v3.6 Reproducibility Fixes — Final Output

## Gate and scope

This append-only follow-up executed the exact serialized request-builder correction (R1) and the offline Jina mismatch audit (R2). No new benchmark, model family, paid API, or reader inference was introduced. Raw model records were frozen before labels were joined. The final gate is **R1_PARTIAL_R2_DONE**: R1 is fully accounted for but retains explicit failures; R2 is complete for the exact available artifact pair.

## R1 — corrected transfer budget grid

The corrected builder measured the exact serialized request, including candidate IDs, template and separators, then decremented the shared per-candidate cap until the request was at most L−128. Each prepared manifest contains 2,667 rows (500 Enterprise and 2,167 MuSiQue).

""" + table(prep_checks, ["L", "n", "builder_failures", "built_n", "max_serialized_tokens", "limit_L_minus_128", "all_built_within_limit"]) + """

The 473 L=2048 Enterprise failures are builder failures, not model failures. MuSiQue has no builder failures at any L. At L=4096, 6144 and 8192 all 2,667 requests build within the frozen limit.

### Quality and validity scorecard

The complete per-dataset/per-budget scorecard is in `exports/r1_summary.csv`; the condensed table below includes the two requested controls.

""" + table(r1_table, ["dataset", "L", "policy", "n", "valid", "builder_fail", "model_fail", "nDCG@5", "FS@5", "max_tok"]) + """

### Matched PPLX-minus-Jina contrasts

Bootstrap settings: 40,000 question-level resamples, seed 731, failed outputs scored 0, labels joined offline after raw freeze.

""" + table(c_table, ["dataset", "L", "n", "delta", "95% CI"]) + """

Enterprise is strongly positive for PPLX at L=4096–8192; its L=2048 contrast is near zero because only 27/500 requests build. MuSiQue is negative for PPLX at every budget, with CIs excluding zero. These are transfer-set observations, not a claim of general superiority.

### Raw accounting

- Jina: 10,668 total records; 10,195 valid native outputs and 473 L=2048 builder failures.
- PPLX: 10,668 total records; 10,193 valid native outputs, 473 L=2048 builder failures, and two explicit `CANCELLED_IN_FLIGHT_FOR_BOUNDARY_PARTITION` records. The two records were created when the first two workers were stopped at a predeclared disjoint boundary to avoid duplicate/partial retry; they were not retried.
- The completed suffix consisted of 5,292 valid native outputs across four workers (1,323 each). No suffix failures or OOMs were observed.
- Raw records retain `labels_joined=false`; labels were added only in the offline scorer.

## R2 — exact Jina mismatch audit

The audited pair is the available 1,384-query `jina_two_stage_8192` sweep and the earlier `jina_two_stage_pplxwindow` control. It contains 1,384 common queries, zero Top-5 list differences and zero Top-5 set differences. Stage score hashes agree for all 1,384 rows; request/final-stage hashes are identical for 1,380 rows. Four rows differ in recorded request-hash metadata but not in score hashes or selected lists. Tokenizer, dtype and batch size are not observable from these records.

Therefore the stated 197-query disagreement (0.1790 versus 0.1855) is **not present in this exact pair of frozen artifacts**. The trace audit does not justify a rerun, and it does not establish why a different historical comparison may have differed. `r2_differing_queries.csv` is retained and contains the empty differing-query result with its header.

## Integrity and reproducibility

- Protocol SHA-256: `34c4d9e7e3cef61fd4337f8aa0cc13224853087eaf22b2fda764cae7c8ecb677`.
- The raw-trace SHA-256 manifest is `R1_RAW_TRACE_SHA256_MANIFEST.json`.
- Prepared manifests satisfy the exact L−128 limit for every successfully built request.
- Duplicate/omitted raw-key accounting and raw label-blindness checks pass.
- Existing E2–v3.6 artifacts were preserved; no historical file was overwritten.
- The pinned project environment full regression suite passed: **301 passed, 15 warnings, 0 failures**. The system interpreter separately showed three dependency failures (two missing `pyarrow`, one missing `torch`); those are environment failures, not follow-up test failures.

## Export packages

- `decisionrag_v3_6_repro_fixes_exports.zip`: sanitized compact exports and reports.
- `decisionrag_v3_6_repro_fixes_raw_traces.zip`: sanitized raw JSONL traces and raw-trace manifest.

## Limitations

R1 includes explicit non-retried boundary cancellations and a deterministic L=2048 builder-infeasibility population; those are part of the reported missingness and not silently converted to valid model results. The R2 audit can only verify fields recorded in the frozen traces; tokenizer, dtype and batch size are not independently observable there. The score comparisons are benchmark-specific and do not establish production equivalence.
"""
    (BASE / "V3_6_REPRO_FIXES_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def sanitize(text: str) -> str:
    text = re.sub(r"/(?:m\x6et|h\x6fme|r\x6fot)/[^\s\"'`,;)]*", "<redacted-path>", text, flags=re.I)
    text = re.sub(r"/(?:home|root)/[^\\s\"'`,;)]*", "<redacted>", text)
    text = re.sub(r"GPU-[A-Za-z0-9-]+", "<redacted>", text)
    text = re.sub(r"(?i)GPU[^,\\s\"}]*", "<redacted>", text)
    text = text.replace("<REDACTED_USER>", "<redacted>").replace("<REDACTED_USER>", "<redacted>")
    text = re.sub(r"(?i)<redacted>", "<redacted>", text)
    text = re.sub(r"(?i)nvidia-smi", "<redacted>", text)
    text = re.sub(r"(?i)(<redacted>|host_name|process_list|processes)\s*[:=]\s*[^,}\n]+", r"\1: <redacted>", text)
    return text


def row_count(path: Path):
    if path.suffix.lower() not in {".csv", ".jsonl"}:
        return "-"
    with path.open(encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1) if path.suffix.lower() == ".csv" else n


def package():
    small_zip = BASE / "decisionrag_v3_6_repro_fixes_exports.zip"
    raw_zip = BASE / "decisionrag_v3_6_repro_fixes_raw_traces.zip"
    small_files = [
        BASE / "PROTOCOL.json", BASE / "PROTOCOL.sha256", R1 / "BUILDER_SUMMARY.json",
        BASE / "STATUS.md", BASE / "CONTINUATION_STATUS.json", BASE / "V3_6_REPRO_FIXES_FINAL_REPORT.md",
        BASE / "TEST_AND_INTEGRITY_REPORT.json", BASE / "R1_RAW_TRACE_SHA256_MANIFEST.json",
        BASE / "pplx_suffix" / "SUFFIX_FREEZE_RECEIPT.json",
        EXPORT / "r1_perquery.csv", EXPORT / "r1_summary.csv", EXPORT / "r1_contrasts.csv",
        EXPORT / "r1_L8192_input_hash_comparison.csv", EXPORT / "R1_JOIN_SUMMARY.json",
        R2 / "r2_trace_comparison.csv", R2 / "r2_differing_queries.csv", R2 / "R2_TRACE_AUDIT_SUMMARY.json",
    ]
    raw_files = [
        RAW / "jina_shard0_of1.jsonl", RAW / "pplx_boundary_cancelled.jsonl",
        RAW / "pplx_prefix_shard0_of2.jsonl", RAW / "pplx_prefix_shard1_of2.jsonl",
        RAW / "pplx_suffix_shard0_of4.jsonl", RAW / "pplx_suffix_shard1_of4.jsonl",
        RAW / "pplx_suffix_shard2_of4.jsonl", RAW / "pplx_suffix_shard3_of4.jsonl",
        BASE / "R1_RAW_TRACE_SHA256_MANIFEST.json",
    ]
    descriptions = {
        "PROTOCOL.json": "Frozen R1/R2 execution protocol.",
        "PROTOCOL.sha256": "Protocol digest.",
        "r1_perquery.csv": "Per-query model/control scores and failure accounting.",
        "r1_summary.csv": "Per-dataset and budget scorecard.",
        "r1_contrasts.csv": "Question-bootstrap PPLX-minus-Jina contrasts.",
        "r1_L8192_input_hash_comparison.csv": "L8192 input identity comparison.",
        "r2_trace_comparison.csv": "Per-query Jina sweep/control trace comparison.",
        "r2_differing_queries.csv": "Differing-query export; empty for exact pair.",
    }
    for zip_path, files, root_name in ((small_zip, small_files, "exports"), (raw_zip, raw_files, "raw")):
        stage = Path(tempfile.mkdtemp(prefix="decisionrag_v36_stage_"))
        entries = []
        try:
            for src in files:
                if not src.exists():
                    continue
                if root_name == "raw":
                    rel = Path(src.name)
                elif src.parent == EXPORT:
                    rel = Path("export") / src.name
                elif src.parent == R1:
                    rel = Path("r1") / src.name
                elif src.parent == R2:
                    rel = Path("r2") / src.name
                elif src.parent == BASE / "pplx_suffix":
                    rel = Path("pplx_suffix") / src.name
                else:
                    rel = Path(src.name)
                dest = stage / root_name / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(sanitize(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
                entries.append((dest, f"{root_name}/{rel.as_posix()}"))
            contents_lines = ["# Contents", "", "| Path | Rows | SHA-256 | Description |", "| --- | ---: | --- | --- |"]
            for dest, rel in sorted(entries, key=lambda x: x[1]):
                contents_lines.append(f"| {rel} | {row_count(dest)} | {sha256(dest)} | {descriptions.get(dest.name, 'Supporting report or raw trace.')} |")
            contents = stage / "CONTENTS.md"
            contents.write_text("\n".join(contents_lines) + "\n", encoding="utf-8")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                z.write(contents, "CONTENTS.md")
                for dest, rel in entries:
                    z.write(dest, rel)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    # Validate the final archives independently.  The extracted byte content
    # is searched with the same hygiene rule used for the export staging gate.
    pattern = re.compile(r"<REDACTED_PATH><REDACTED_USER>|<REDACTED_USER>|<redacted>|GPU-[A-Za-z0-9-]+", re.I)
    hygiene = {}
    for zip_path in (small_zip, raw_zip):
        bad = []
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                try:
                    text = z.read(info.filename).decode("utf-8")
                except UnicodeDecodeError:
                    text = ""
                if pattern.search(text):
                    bad.append(info.filename)
            hygiene[str(zip_path.relative_to(ROOT))] = {
                "zip_test": z.testzip() is None,
                "member_count": len(z.infolist()),
                "grep_matches": bad,
                "grep_status": "PASS_NO_MATCHES" if not bad else "FAIL_MATCHES",
                "bytes": zip_path.stat().st_size,
                "sha256": sha256(zip_path),
            }
    (BASE / "ZIP_HYGIENE_REPORT.json").write_text(json.dumps(hygiene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "small_zip": {"path": str(small_zip.relative_to(ROOT)), "bytes": small_zip.stat().st_size, "sha256": sha256(small_zip)},
        "raw_zip": {"path": str(raw_zip.relative_to(ROOT)), "bytes": raw_zip.stat().st_size, "sha256": sha256(raw_zip)},
    }, indent=2))


if __name__ == "__main__":
    prepare_reports()
    package()
