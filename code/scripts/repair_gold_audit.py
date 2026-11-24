#!/usr/bin/env python3
"""Repair the M1 provenance audit to the SDD's 20 examples per dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpot", "musique", "wiki2")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    questions = load_jsonl(ROOT / "data/manifests/questions.jsonl")
    audit = []
    for dataset in DATASETS:
        selected = sorted((row for row in questions if row["dataset"] == dataset), key=lambda row: row["query_id"])[:20]
        if len(selected) != 20 or not all(row["gold_mapping_valid"] for row in selected):
            raise RuntimeError(f"audit selection invalid for {dataset}")
        for row in selected:
            audit.append({
                "dataset": row["dataset"],
                "query_id": row["query_id"],
                "question": row["question"],
                "answer": row["answer"],
                "gold_doc_ids": row["gold_doc_ids"],
                "gold_mapping_valid": row["gold_mapping_valid"],
                "mapping_ambiguities": row["mapping_ambiguities"],
                "context_doc_ids": row["context_doc_ids"],
                "source": row["provenance"],
                "audit_selection": "first_sorted_selected_20_per_dataset",
            })
    path = ROOT / "data/manifests/gold_mapping_audit.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audit), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = ROOT / "data/manifests/m1_data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["gold_audit"] = digest
    manifest["mapping"]["audit_rows"] = len(audit)
    manifest["gold_audit_selection"] = "20 valid selected records per dataset; first sorted by query_id; repair-only provenance update"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit_rows": len(audit), "counts": {dataset: sum(row["dataset"] == dataset for row in audit) for dataset in DATASETS}, "gold_audit_sha256": digest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
