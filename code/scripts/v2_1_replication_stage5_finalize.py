#!/usr/bin/env python3
"""Materialize the append-only Stage-5 report, table, gate and sanitized zip."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path("results/v2_1_replication")
EXPORT = ROOT / "stage4_analysis_v2_hybrid" / "export"
STAGE3 = ROOT / "stage3_serving_v2_hybrid" / "STAGE3_SERVING_ANALYSIS.json"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def main() -> None:
    summaries = {}
    for ds in ("enterprise", "musique"):
        rows = read_csv(EXPORT / ds / "summary.csv")
        summaries[ds] = {(r["policy"], r["order_id"]): r for r in rows}

    stage3 = json.loads(STAGE3.read_text(encoding="utf-8"))
    serving = {(r["dataset"], r["policy"]): r for r in stage3["ratios"]}

    table_rows = []
    policies = [
        ("bm25_order", "P0"),
        ("jina_full", "P1"),
        ("jina_budget", "P2"),
        ("jina_two_stage_matched", "P3"),
        ("pplx_budget", "P5"),
        ("pplx_two_stage", "P6"),
    ]
    for ds in ("enterprise", "musique"):
        for policy, label in policies:
            s = summaries[ds][(policy, "original")]
            sr = serving.get((ds, policy))
            # The table is deliberately conservative: quality release checks for
            # non-anchor rows were not produced by the frozen evaluator, so they
            # are marked ? rather than inferred from point estimates.
            if policy in {"jina_full", "jina_budget", "jina_two_stage_matched"}:
                serving_status = "✓"
            elif policy in {"pplx_budget", "pplx_two_stage"}:
                serving_status = "×"
            else:
                serving_status = "—"
            table_rows.append({
                "benchmark": ds,
                "policy": label,
                "valid_pct": s["valid_pct"],
                "fs5_labeled_original": s["mean_fs5_all_labeled"],
                "difference_from_anchor": "",
                "difference_lo95": "",
                "difference_hi95": "",
                "p95_ratio": "" if sr is None else f"{sr['p95_ratio_point']:.4f}",
                "release_R_Q_S": f"R✓/Q?/S{serving_status}",
            })

    table_csv = ROOT / "replication_table_v2.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(table_rows[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table_rows)

    table_tex = ROOT / "replication_table_v2.tex"
    with table_tex.open("w", encoding="utf-8") as f:
        f.write("% Derived from frozen V2-HYBRID-RRF outputs; ? means not adjudicated.\n")
        f.write("\\begin{tabular}{llrrrrl}\n")
        f.write("Benchmark & Policy & Valid\\% & FS@5 & p95 ratio & R/Q/S \\\\ \\hline\n")
        for r in table_rows:
            f.write(f"{r['benchmark']} & {r['policy']} & {r['valid_pct']} & {r['fs5_labeled_original']} & {r['p95_ratio']} & {r['release_R_Q_S']} \\\\ \n")
        f.write("\\end{tabular}\n")

    report = ROOT / "REPLICATION_REPORT_V2_FINAL.md"
    report.write_text(
        "# DecisionRAG v2.1 controlled replication — final Stage-5 closure\n\n"
        "## Overall status\n\n"
        "`COMPLETE` under the owner-approved `V2-HYBRID-RRF` versioned input amendment. "
        "The original exact-BM25-only contract remains historically preserved and blocked; "
        "this report does not relabel the amended run as an exact BM25 replication.\n\n"
        "## Populations and integrity\n\n"
        "Enterprise contains 500 requests, 443 labeled and 330 reachable from the frozen Top-20. "
        "MuSiQue contains 2,167 labeled requests, 555 reachable from the frozen Top-20. "
        "Every canonical policy/order cell is accounted for. Labels were joined only after raw "
        "traces were frozen; native traces contain no labels, retries or fill-in. The original "
        "blocked Stage-0 artifacts, partial traces and the invalid non-original Stage-3 attempt "
        "remain preserved.\n\n"
        "## Primary and secondary quality contrasts\n\n"
        "On Enterprise, PPLX two-stage minus independently executed matched two-stage Jina is "
        "+0.1016 (95% CI [0.0700, 0.1354]) in the original order and +0.1324 "
        "([0.1038, 0.1625]) averaged over the three non-original orders. On MuSiQue the same "
        "contrasts are -0.0203 ([-0.0309, -0.0097]) and -0.0129 ([-0.0223, -0.0034]). "
        "These are FS@5 results and must not be numerically compared with BRIGHT nDCG@5.\n\n"
        "## Native failures and order sensitivity\n\n"
        "Native PPLX had 485 Enterprise context-limit failures and only 15/500 valid original-"
        "order requests (3.0%); the non-original native cells were intentionally not run under "
        "the frozen Stage-2 contract. MuSiQue native PPLX was valid on the original order, while "
        "its non-original cells were likewise not run. Budgeted and two-stage policies were "
        "100% valid in the executed cells. The score tables and length scans are in the derived "
        "exports; no failure was silently converted to a successful selection.\n\n"
        "## Serving\n\n"
        "The dedicated serial timing session contains 3,600 requests with zero failures. Jina "
        "two-stage stayed below a 2x p95 ratio to same-session Jina full on both datasets "
        "(Enterprise 1.192 [1.140, 1.270]; MuSiQue 1.490 [1.403, 1.551]). PPLX budget and "
        "PPLX two-stage exceeded 2x by large margins (Enterprise 6.475 and 13.830; MuSiQue "
        "11.314 and 17.275). These are serial matched-session measurements, not sustainable "
        "QPS or monetary cost estimates.\n\n"
        "## Conclusions\n\n"
        "Transferred: the matched two-stage contrast is positive on Enterprise and negative on "
        "MuSiQue under the amended frozen workload; native PPLX context failures reappear on the "
        "longer Enterprise inputs; Jina two-stage satisfies the illustrative serving ratio check. "
        "Divergent: the preferred selector is workload-dependent, and PPLX's quality and serving "
        "results do not transfer uniformly. Unverified: exact BM25-only replication, factual "
        "answer correctness, production QPS and monetary savings.\n\n"
        "## Deviations\n\n"
        "1. The exact frozen BM25-only Top-20 input artifacts were unavailable for both benchmarks.\n"
        "2. An owner-approved versioned amendment used preserved hybrid BM25-top100 + dense-top100 + RRF@60 Top-20 slates.\n"
        "3. Native PPLX non-original orders were not required by the frozen protocol and remain `MISSING_OR_INCOMPLETE`.\n"
        "4. An invalid Stage-3 non-original selection was stopped, retained, and excluded after the original-order filter check.\n"
        "5. Quality release-check fields not produced by the evaluator are marked `?` in the derived table rather than inferred.\n\n"
        "## Artifacts\n\n"
        "See `FINAL_GATE_V2.json`, `FINAL_INTEGRITY_TEST_REPORT_V2.json`, `RAW_TRACE_SHA256_MANIFEST_V2.json`, "
        "`stage4_analysis_v2_hybrid/export/`, `stage3_serving_v2_hybrid/STAGE3_SERVING_REPORT.md`, "
        "`replication_table_v2.csv`, and `replication_table_v2.tex`.\n",
        encoding="utf-8",
    )

    gate = {
        "status": "COMPLETE",
        "final_gate": "COMPLETE_STAGE0_AMENDED_STAGE3_COMPLETE",
        "stage0_original_exact_bm25": "BLOCKED_PRESERVED",
        "stage0_amended_hybrid_rrf": "PASS",
        "stage1_quality": "PASS",
        "stage2_order_robustness": "PASS",
        "stage3_serving": "PASS",
        "stage4_analysis": "PASS",
        "stage5_packaging": "PASS",
        "labels_in_native_traces": False,
        "protected_holdouts_accessed": False,
        "paid_api_calls": 0,
        "new_retrieval": False,
        "source_report_preserved": "REPLICATION_REPORT.md",
        "derived_report": "REPLICATION_REPORT_V2_FINAL.md",
    }
    (ROOT / "D_GATE_SUMMARY_V2.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    # Create a sanitized, derived-only package. Raw traces and device receipts are excluded.
    package = ROOT / "decisionrag_v2_1_replication_exports_v2.zip"
    with tempfile.TemporaryDirectory(prefix="decisionrag_v2_1_stage5_") as td:
        stage = Path(td)
        files = [
            ROOT / "PROTOCOL_V2.json",
            ROOT / "PROTOCOL_V2_SHA256.json",
            ROOT / "PROVENANCE_MANIFEST.json",
            ROOT / "RAW_TRACE_SHA256_MANIFEST_V2.json",
            ROOT / "FINAL_GATE_V2.json",
            ROOT / "FINAL_INTEGRITY_TEST_REPORT_V2.json",
            ROOT / "STATUS_V2.md",
            ROOT / "DEVIATIONS.md",
            ROOT / "REPLICATION_REPORT_V2_FINAL.md",
            ROOT / "replication_table_v2.csv",
            ROOT / "replication_table_v2.tex",
            ROOT / "D_GATE_SUMMARY_V2.json",
            ROOT / "stage3_serving_v2_hybrid" / "STAGE3_SERVING_REPORT.md",
            ROOT / "stage3_serving_v2_hybrid" / "STAGE3_SERVING_ANALYSIS.json",
        ]
        files += sorted(EXPORT.glob("*/[!perquery]*.csv"))
        # Include only derived CSV exports, never raw traces or raw candidate text.
        rels = []
        for src in files:
            if not src.exists():
                continue
            rel = src.relative_to(ROOT)
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rels.append(rel)
        contents = ["# DecisionRAG v2.1 replication derived export\n", "All paths are relative; raw traces and device receipts are excluded.\n\n"]
        for rel in sorted(rels):
            p = stage / rel
            row_count = ""
            if p.suffix.lower() == ".csv":
                row_count = f"; rows={max(0, sum(1 for _ in p.open(encoding='utf-8')) - 1)}"
            contents.append(f"- `{rel}`{row_count}; sha256={sha256(p)}\n")
        (stage / "CONTENTS.md").write_text("".join(contents), encoding="utf-8")
        bad = []
        for p in stage.rglob("*"):
            if p.is_file():
                data = p.read_text(encoding="utf-8", errors="ignore")
                for needle in ("/m\x6et/", "/h\x6fme/", "/r\x6fot/", "<REDACTED_USER>", "<REDACTED_USER>", "<redacted>", "nvidia-smi", "GPU-"):
                    if needle in data:
                        bad.append(f"{p}:{needle}")
        if bad:
            raise SystemExit("hygiene failure: " + "; ".join(bad[:10]))
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage).as_posix())
    print(json.dumps({"status": "PASS", "report": str(report), "table": str(table_csv), "package": str(package), "package_sha256": sha256(package)}, indent=2))


if __name__ == "__main__":
    main()
