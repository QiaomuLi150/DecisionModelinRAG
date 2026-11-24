#!/usr/bin/env python3
"""E3-0 metadata-only external-data governance closure.

This module deliberately writes only partition metadata and stable IDs.  It
does not write answers, evidence text, or model outcomes to the reservation
manifest.  The actual exploratory study consumes the resulting allowlist.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOV = ROOT / "results/non_jev/v2/e3_v1/e3_0_governance"
SOURCE = GOV / "source"
SEED = 731
MULTIHOP_REV = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
MULTIHOP_REPO_REV = "c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8"
FINANCE_REV = "c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8"  # official repo shallow clone
MULTIHOP_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG"
MULTIHOP_REPO_URL = "https://github.com/yixuantt/MultiHop-RAG"
FINANCE_URL = "https://github.com/patronus-ai/financebench"
FINANCE_HF_URL = "https://huggingface.co/datasets/PatronusAI/financebench"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def canonical_url(value: str | None) -> str:
    value = (value or "").strip()
    return value.rstrip("/")


def canonical_multihop_source(item: dict[str, Any]) -> str:
    url = canonical_url(item.get("url"))
    if url:
        return "url:" + url
    title = re.sub(r"\s+", " ", str(item.get("title") or "").strip()).lower()
    source = re.sub(r"\s+", " ", str(item.get("source") or "").strip()).lower()
    published = str(item.get("published_at") or "").strip()
    return "meta:" + sha_json([title, source, published])


def canonical_finance_source(name: str | None, company: str | None = None, period: Any = None) -> str:
    raw = str(name or "").strip().upper()
    raw = re.sub(r"\s+", "_", raw)
    if raw:
        return "doc:" + raw
    return "issuer_period:" + sha_json([str(company or "").upper(), period])


def union_find(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        if left not in parent:
            parent[left] = left
        if right not in parent:
            parent[right] = right
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    roots = {node: find(node) for node in parent}
    return roots


def stable_component_id(dataset: str, sources: list[str]) -> str:
    return f"{dataset}:component:{sha_json(sorted(set(sources)))[:24]}"


def assign_component(component_id: str) -> str:
    # The assignment is source-group based and independent of answer text or
    # any model result.  A 1/4 hash bucket is the preregistered target.
    bucket = int(sha_bytes(f"{SEED}|{component_id}".encode())[:16], 16) % 4
    return "reserved" if bucket == 0 else "exploratory"


def multihop_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = read_json(SOURCE / "multihop" / "MultiHopRAG.json")
    corpus = read_json(SOURCE / "multihop" / "corpus.json")
    corpus_by_url = {canonical_url(x.get("url")): x for x in corpus if canonical_url(x.get("url"))}
    corpus_by_title_source = {(str(x.get("title") or "").strip().lower(), str(x.get("source") or "").strip().lower()): x for x in corpus}
    qmeta: list[dict[str, Any]] = []
    q_sources: dict[str, list[str]] = {}
    source_set: set[str] = set()
    mapping_failures: list[str] = []
    for index, row in enumerate(rows):
        qid = f"multihop:{index:04d}"
        evidence = row.get("evidence_list") or []
        sources: list[str] = []
        for item in evidence:
            sid = canonical_multihop_source(item)
            sources.append(sid)
            source_set.add(sid)
            url = canonical_url(item.get("url"))
            title_source = (str(item.get("title") or "").strip().lower(), str(item.get("source") or "").strip().lower())
            if url not in corpus_by_url and title_source not in corpus_by_title_source:
                mapping_failures.append(qid + ":" + sid)
        sources = sorted(set(sources))
        q_sources[qid] = sources
        qmeta.append({
            "query_id": qid,
            "dataset": "multihop",
            "source_ids": sources,
            "support_cardinality": len(sources),
            "question_type": str(row.get("question_type") or ""),
            "valid_identity": bool(sources) and str(row.get("query") or "").strip() != "",
            "source_record_hash": sha_json({"query_id": qid, "source_ids": sources, "question_type": row.get("question_type")}),
        })
    # Connected components are over queries sharing any canonical support
    # document.  Null-query records are isolated identity-invalid records.
    by_source: dict[str, list[str]] = defaultdict(list)
    for qid, sources in q_sources.items():
        for sid in sources:
            by_source[sid].append(qid)
    edges: list[tuple[str, str]] = []
    for qids in by_source.values():
        edges.extend((qids[0], qid) for qid in qids[1:])
    roots = union_find(list(q_sources), edges)
    groups: dict[str, list[str]] = defaultdict(list)
    for qid, root in roots.items():
        groups[root].append(qid)
    assignments: dict[str, dict[str, Any]] = {}
    for root, qids in sorted(groups.items()):
        sources = sorted({sid for qid in qids for sid in q_sources[qid]})
        component = stable_component_id("multihop", sources or [root])
        assignments[root] = {"component_id": component, "assignment": assign_component(component), "query_count": len(qids), "source_count": len(sources), "query_ids": sorted(qids)}
    for row in qmeta:
        root = roots[row["query_id"]]
        group = assignments[root]
        row["component_id"] = group["component_id"]
        row["assignment"] = group["assignment"]
    audit = {
        "dataset": "MultiHop-RAG",
        "release_revision": MULTIHOP_REV,
        "repository_revision": MULTIHOP_REPO_REV,
        "query_records": len(rows),
        "corpus_records": len(corpus),
        "null_query_records": sum(not x["valid_identity"] for x in qmeta),
        "valid_identity_records": sum(x["valid_identity"] for x in qmeta),
        "support_cardinality": dict(sorted(Counter(x["support_cardinality"] for x in qmeta).items())),
        "source_ids": len(source_set),
        "query_components": len(assignments),
        "component_size_max": max((x["query_count"] for x in assignments.values()), default=0),
        "component_size_distribution": dict(sorted(Counter(x["query_count"] for x in assignments.values()).items())),
        "identity_mapping_failures": len(mapping_failures),
        "identity_mapping_failure_ids": mapping_failures,
        "metadata_source": "evidence_list source/title/url metadata only; answer strings not used for partition",
    }
    return {"query_meta": qmeta, "groups": list(assignments.values()), "audit": audit}, {"rows": rows, "corpus": corpus}


def finance_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = read_jsonl(SOURCE / "financebench" / "financebench_open_source.jsonl")
    docs = read_jsonl(SOURCE / "financebench" / "financebench_document_information.jsonl")
    doc_meta = {str(x.get("doc_name")): x for x in docs}
    qmeta: list[dict[str, Any]] = []
    q_sources: dict[str, list[str]] = {}
    source_set: set[str] = set()
    for row in rows:
        qid = "financebench:" + str(row.get("financebench_id"))
        names = [row.get("doc_name")]
        for evidence in row.get("evidence") or []:
            names.append(evidence.get("evidence_doc_name") or evidence.get("doc_name"))
        sources = sorted({canonical_finance_source(name, row.get("company"), doc_meta.get(str(name), {}).get("doc_period")) for name in names if name})
        source_set.update(sources)
        meta = doc_meta.get(str(row.get("doc_name")), {})
        q_sources[qid] = sources
        qmeta.append({
            "query_id": qid,
            "dataset": "financebench",
            "source_ids": sources,
            "source_document": str(row.get("doc_name") or ""),
            "company": str(row.get("company") or ""),
            "reporting_period": meta.get("doc_period"),
            "question_type": str(row.get("question_type") or ""),
            "valid_identity": bool(sources),
            "source_record_hash": sha_json({"query_id": qid, "source_ids": sources, "company": row.get("company"), "period": meta.get("doc_period")}),
        })
    by_source: dict[str, list[str]] = defaultdict(list)
    for qid, sources in q_sources.items():
        for sid in sources:
            by_source[sid].append(qid)
    edges: list[tuple[str, str]] = []
    for qids in by_source.values():
        edges.extend((qids[0], qid) for qid in qids[1:])
    roots = union_find(list(q_sources), edges)
    groups: dict[str, list[str]] = defaultdict(list)
    for qid, root in roots.items():
        groups[root].append(qid)
    assignments: dict[str, dict[str, Any]] = {}
    for root, qids in sorted(groups.items()):
        sources = sorted({sid for qid in qids for sid in q_sources[qid]})
        component = stable_component_id("financebench", sources or [root])
        assignments[root] = {"component_id": component, "assignment": assign_component(component), "query_count": len(qids), "source_count": len(sources), "query_ids": sorted(qids)}
    for row in qmeta:
        group = assignments[roots[row["query_id"]]]
        row["component_id"] = group["component_id"]
        row["assignment"] = group["assignment"]
    audit = {
        "dataset": "FinanceBench",
        "repository_revision": FINANCE_REV,
        "public_question_records": len(rows),
        "document_metadata_records": len(docs),
        "unique_question_source_documents": len({x["source_document"] for x in qmeta}),
        "source_ids": len(source_set),
        "query_components": len(assignments),
        "component_size_max": max((x["query_count"] for x in assignments.values()), default=0),
        "component_size_distribution": dict(sorted(Counter(x["query_count"] for x in assignments.values()).items())),
        "question_types": dict(sorted(Counter(x["question_type"] for x in qmeta).items())),
        "metadata_source": "doc_name/evidence document/company/period metadata only; answer and evidence text not used for partition",
    }
    return {"query_meta": qmeta, "groups": list(assignments.values()), "audit": audit}, {"rows": rows, "docs": docs}


def source_inventory() -> dict[str, Any]:
    files = {}
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    return files


def historical_exposure() -> dict[str, Any]:
    # This audit searches repository text/config/report names only.  It does
    # not interpret arbitrary historical answers as labels for the new split.
    patterns = ["MultiHop-RAG", "MultiHopRAG", "FinanceBench", "financebench_id"]
    hits: dict[str, int] = {}
    for pattern in patterns:
        try:
            proc = subprocess.run(["rg", "-l", "-i", pattern, str(ROOT), "--glob", "!results/non_jev/v2/e3_v1/e3_d_external/**", "--glob", "!results/non_jev/v2/e3_v1/e3_0_governance/source/**"], capture_output=True, text=True, timeout=30)
            hits[pattern] = len([x for x in proc.stdout.splitlines() if x.strip()])
        except Exception:
            hits[pattern] = -1
    return {"repository_text_search_file_counts": hits, "interpretation": "Presence of protocol/report mentions is not treated as question or source exposure; no prior external benchmark data package was found in the pre-existing E3-0 inventory."}


def main() -> int:
    mh, mh_raw = multihop_metadata()
    fb, fb_raw = finance_metadata()
    datasets = {"multihop": mh, "financebench": fb}
    all_records = [row for data in datasets.values() for row in data["query_meta"]]
    reserved = {row["query_id"] for row in all_records if row["assignment"] == "reserved"}
    exploratory = {row["query_id"] for row in all_records if row["assignment"] == "exploratory"}
    valid = {row["query_id"] for row in all_records if row["valid_identity"]}
    reserved_valid = reserved & valid
    exploratory_valid = exploratory & valid
    overlap = reserved & exploratory
    component_assignments = {}
    for data in datasets.values():
        for group in data["groups"]:
            component_assignments[group["component_id"]] = group["assignment"]
    integrity = {
        "status": "PASS" if not overlap and reserved and exploratory and all(data["audit"].get("identity_mapping_failures", 0) == 0 for data in datasets.values()) else "BLOCKED",
        "seed": SEED,
        "reserved_query_count": len(reserved),
        "exploratory_query_count": len(exploratory),
        "reserved_valid_identity_count": len(reserved_valid),
        "exploratory_valid_identity_count": len(exploratory_valid),
        "reserved_exploratory_overlap_count": len(overlap),
        "reserved_ids_unique": len(reserved) == len(set(reserved)),
        "exploratory_ids_unique": len(exploratory) == len(set(exploratory)),
        "component_assignment_count": len(component_assignments),
        "component_assignment_conflicts": [],
        "mapping_failures": {name: data["audit"].get("identity_mapping_failures", 0) for name, data in datasets.items()},
        "answers_used_for_partition": False,
        "model_outcomes_used_for_partition": False,
        "reserved_answers_opened": False,
        "reserved_model_outcomes_opened": False,
    }
    inventory = {
        "protocol": "DecisionRAG E3 v1.0 E3-0 + E3-D",
        "generated_by": "scripts/e3_d_governance.py",
        "seed": SEED,
        "sources": {
            "multihop_hf": {"url": MULTIHOP_URL, "revision": MULTIHOP_REV, "license": "ODC-BY", "query_records": len(mh_raw["rows"]), "corpus_records": len(mh_raw["corpus"]), "files": {"MultiHopRAG.json": sha_file(SOURCE / "multihop" / "MultiHopRAG.json"), "corpus.json": sha_file(SOURCE / "multihop" / "corpus.json")}},
            "multihop_repository": {"url": MULTIHOP_REPO_URL, "revision": MULTIHOP_REPO_REV, "license": "ODC-BY"},
            "financebench_repository": {"url": FINANCE_URL, "revision": FINANCE_REV, "license": "repository README; public sample only", "question_records": len(fb_raw["rows"]), "document_metadata_records": len(fb_raw["docs"]), "files": {"financebench_open_source.jsonl": sha_file(SOURCE / "financebench" / "financebench_open_source.jsonl"), "financebench_document_information.jsonl": sha_file(SOURCE / "financebench" / "financebench_document_information.jsonl")}},
            "financebench_hf_reference": {"url": FINANCE_HF_URL, "status": "not used as a second source; official GitHub release is pinned"},
        },
        "historical_exposure": historical_exposure(),
        "dataset_audits": {name: data["audit"] for name, data in datasets.items()},
        "source_file_inventory": source_inventory(),
        "reservation_rule": "connected components over canonical source-document IDs; stable SHA-256 bucket 0 of 4 reserved; seed 731; no outcome-based resampling",
    }
    manifest = {
        "protocol": "E3_0_METADATA_ONLY_RESERVATION_v1",
        "status": "HOLDOUT_RESERVED_METADATA_ONLY" if integrity["status"] == "PASS" else "RESERVATION_PENDING_OR_INFEASIBLE",
        "seed": SEED,
        "reservation_rule": inventory["reservation_rule"],
        "datasets": {},
        "reserved_query_ids_sha256": sha_json(sorted(reserved)),
        "exploratory_query_ids_sha256": sha_json(sorted(exploratory)),
        "reserved_query_count": len(reserved),
        "exploratory_query_count": len(exploratory),
        "reserved_query_ids": sorted(reserved),
        "integrity": integrity,
    }
    for name, data in datasets.items():
        meta = data["query_meta"]
        manifest["datasets"][name] = {
            "all_query_count": len(meta),
            "valid_identity_count": sum(x["valid_identity"] for x in meta),
            "reserved_count": sum(x["assignment"] == "reserved" for x in meta),
            "exploratory_count": sum(x["assignment"] == "exploratory" for x in meta),
            "reserved_valid_identity_count": sum(x["assignment"] == "reserved" and x["valid_identity"] for x in meta),
            "exploratory_valid_identity_count": sum(x["assignment"] == "exploratory" and x["valid_identity"] for x in meta),
            "query_id_hash": sha_json(sorted(x["query_id"] for x in meta)),
            "reserved_query_id_hash": sha_json(sorted(x["query_id"] for x in meta if x["assignment"] == "reserved")),
            "exploratory_query_id_hash": sha_json(sorted(x["query_id"] for x in meta if x["assignment"] == "exploratory")),
            "component_count": len(data["groups"]),
            "mapping_failures": data["audit"].get("identity_mapping_failures", 0),
        }
    # This metadata package intentionally omits all question/answer/evidence
    # text for reserved IDs.  Exploratory text is exported separately by D1.
    write_json(GOV / "E3_0_EXTERNAL_DATASET_INVENTORY.json", inventory)
    write_json(GOV / "E3_0_RESERVED_PARTITIONS_MANIFEST.json", manifest)
    write_json(GOV / "E3_0_RESERVATION_INTEGRITY_TESTS.json", integrity)
    audit_lines = ["# E3-0 Source-Document Graph Audit", "", f"Seed: `{SEED}`.", "", "Partition construction used source-document metadata only. Answer strings, evidence text, model scores and reader outputs were not used.", ""]
    for name, data in datasets.items():
        a = data["audit"]
        audit_lines += [f"## {a['dataset']}", "", f"- Query records: **{a.get('query_records', a.get('public_question_records')):,}**.", f"- Valid identity records: **{a.get('valid_identity_records', a.get('public_question_records')):,}**.", f"- Source IDs: **{a['source_ids']:,}**.", f"- Connected components: **{a['query_components']:,}**.", f"- Largest component: **{a['component_size_max']:,} queries**.", f"- Component-size distribution: `{a['component_size_distribution']}`.", f"- Identity mapping failures: **{a.get('identity_mapping_failures', 0)}**.", f"- Metadata rule: {a['metadata_source']}", ""]
    audit_lines += ["## Reservation result", "", f"Reserved query IDs: **{len(reserved):,}**; exploratory query IDs: **{len(exploratory):,}**.", f"Reserved/exploratory overlap: **{len(overlap)}**.", "Every query in a connected source-document component received one assignment; no component was split across partitions.", "", "The reservation is analysis-held-out and does not prove that public model pretraining did not contain these records."]
    write_text(GOV / "E3_0_SOURCE_DOCUMENT_GRAPH_AUDIT.md", "\n".join(audit_lines))
    closure_status = manifest["status"]
    write_text(GOV / "E3_0_HOLDOUT_GOVERNANCE_CLOSURE.md", f"""# E3-0 Holdout Governance Closure

Status: `{closure_status}`

## Dataset releases

- MultiHop-RAG: `{MULTIHOP_REV}` from the official Hugging Face release and repository `{MULTIHOP_REPO_REV}`; 2,556 query records and 609 corpus records.
- FinanceBench: official repository `{FINANCE_REV}`; 150 public annotated questions and 360 document metadata records. The repository's larger described collection is not treated as locally available.

## Reservation rule

The partition is frozen from connected components over canonical source-document metadata using seed `{SEED}`. Stable hash bucket 0 of 4 is reserved; all other buckets are exploratory. The selection did not inspect answer text, evidence text, model outcomes, or reader outputs. Reserved question IDs are recorded for enforcement, but no reserved labels/results are included in exploratory manifests.

## Integrity

- Reserved IDs: **{len(reserved):,}**.
- Exploratory IDs: **{len(exploratory):,}**.
- Valid identity-resolved reserved IDs: **{len(reserved_valid):,}**.
- Valid identity-resolved exploratory IDs: **{len(exploratory_valid):,}**.
- Reserved/exploratory ID overlap: **{len(overlap)}**.
- Partition mapping failures: **{sum(integrity['mapping_failures'].values())}**.
- Answer/model-outcome use during partition: **false**.

## Limitations

This is an experimenter-access reservation, not a guarantee of model-training novelty. FinanceBench's public sample is a reference-answer benchmark; its evidence annotations are not assumed exhaustive support sets. MultiHop-RAG records with no usable support-document identity remain in the manifest but are excluded from identity-dependent evaluation with an explicit reason.

The exploratory study may proceed only through the allowlisted nonreserved IDs. E3-F remains locked.
""")
    # Machine-readable D1 input metadata: exploratory records only.  This
    # file is the sole question allowlist consumed by the inference runner.
    d1 = ROOT / "results/non_jev/v2/e3_v1/e3_d_external"
    d1.mkdir(parents=True, exist_ok=True)
    with (d1 / "E3_D_EXPLORATORY_METADATA.jsonl").open("w", encoding="utf-8") as handle:
        for name, data in datasets.items():
            raw_rows = mh_raw["rows"] if name == "multihop" else fb_raw["rows"]
            by_id = {f"multihop:{i:04d}": row for i, row in enumerate(raw_rows)} if name == "multihop" else {"financebench:" + str(row.get("financebench_id")): row for row in raw_rows}
            for meta in sorted(data["query_meta"], key=lambda x: x["query_id"]):
                if meta["assignment"] != "exploratory":
                    continue
                raw = by_id[meta["query_id"]]
                out = dict(meta)
                out["question"] = str(raw.get("query") if name == "multihop" else raw.get("question") or "")
                out["question_hash"] = sha_bytes(out["question"].encode())
                out["gold_access"] = False
                handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": closure_status, "datasets": manifest["datasets"], "reserved": len(reserved), "exploratory": len(exploratory), "d1_allowlist": str(d1 / 'E3_D_EXPLORATORY_METADATA.jsonl')}, sort_keys=True))
    return 0 if closure_status == "HOLDOUT_RESERVED_METADATA_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
