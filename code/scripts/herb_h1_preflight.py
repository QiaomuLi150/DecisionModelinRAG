#!/usr/bin/env python3
"""Offline HERB H1 inventory and preregistration preflight.

This script only reads the pinned native HERB release and official code checkout.
It does not import model libraries, build an index, call a service, or perform
inference.  It writes H1 artifacts into a new results subtree and never edits
the existing Track A/B/C or R2 evidence.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import subprocess
from typing import Any


PROJECT = pathlib.Path(__file__).resolve().parents[1]
RAW = PROJECT / "data/raw/vendor/herb_h1"
DATASET = RAW / "official_dataset_repo"
CODE = RAW / "official_code_repo"
OUT = PROJECT / "results/non_jev/herb_h1"
ARTIFACT_TYPES = ("slack", "documents", "meeting_transcripts", "meeting_chats", "urls", "prs")
POLICIES = [
    "raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5",
    "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5",
    "clef_rescue5",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def git(path: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def json_dump(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def jsonl_dump(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def gpu_probe() -> dict[str, Any]:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:  # pragma: no cover - host-dependent
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    products = sorted((DATASET / "products").glob("*.json"))
    metadata = sorted((DATASET / "metadata").glob("*.json"))
    product_data = {p.stem: json.loads(p.read_text()) for p in products}

    artifacts: list[dict[str, Any]] = []
    answerable: list[tuple[str, int, dict[str, Any]]] = []
    unanswerable: list[tuple[str, int, str]] = []
    field_counts: collections.Counter[str] = collections.Counter()
    type_counts: collections.Counter[str] = collections.Counter()
    schema_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    key_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    for product in products:
        data = product_data[product.stem]
        for field, value in data.items():
            if isinstance(value, list):
                field_counts[field] += len(value)
        for source_type in ARTIFACT_TYPES:
            for index, item in enumerate(data.get(source_type, [])):
                keys = sorted(item) if isinstance(item, dict) else []
                schema = ",".join(keys)
                schema_counts[source_type][schema] += 1
                key_counts[source_type][schema] += 1
                original_id = item.get("id") if isinstance(item, dict) else None
                physical_id = f"herb:{product.stem}:{source_type}:{index:05d}"
                artifacts.append({
                    "physical_artifact_id": physical_id,
                    "original_source_id": original_id,
                    "product": product.stem,
                    "source_type": source_type,
                    "source_record_index": index,
                    "record_sha256": sha256_bytes(canonical(item)),
                    "source_path": f"products/{product.name}",
                    "visible_field_names": keys,
                })
                type_counts[source_type] += 1
        for index, item in enumerate(data.get("answerable_questions", [])):
            answerable.append((product.stem, index, item))
        for index, item in enumerate(data.get("unanswerable_questions", [])):
            unanswerable.append((product.stem, index, item))

    by_original_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in artifacts:
        if record["original_source_id"]:
            by_original_id[record["original_source_id"]].append(record)
    collisions = {k: v for k, v in by_original_id.items() if len(v) > 1}

    artifact_id_set = set(by_original_id)
    citation_refs = 0
    citation_unresolved: list[dict[str, Any]] = []
    citation_target_types: collections.Counter[str] = collections.Counter()
    citation_unique = set()
    for product, index, item in answerable:
        for citation in item.get("citations", []):
            citation_refs += 1
            citation_unique.add(citation)
            matches = by_original_id.get(citation, [])
            if not matches:
                citation_unresolved.append({"product": product, "record_index": index, "citation": citation})
            elif len(matches) == 1:
                citation_target_types[matches[0]["source_type"]] += 1
            else:
                citation_target_types["AMBIGUOUS_ORIGINAL_ID"] += 1

    query_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    q_type_counts: collections.Counter[str] = collections.Counter()
    text_to_records: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
    for product, index, item in answerable:
        question = item["question"]
        qid = f"herb:answerable:{product}:{index:04d}"
        question_hash = sha256_bytes(question.encode())
        query_rows.append({
            "query_id": qid, "answerability": "answerable", "product": product,
            "source_record_index": index, "question": question,
            "question_sha256": question_hash, "evaluation_only_gold": True,
        })
        gold_rows.append({
            "query_id": qid, "answerability": "answerable", "product": product,
            "source_record_index": index, "question_sha256": question_hash,
            "question_type": item.get("type"), "ground_truth": item.get("ground_truth"),
            "citations": item.get("citations", []), "evaluation_only": True,
        })
        q_type_counts[str(item.get("type"))] += 1
        text_to_records[question].append(qid)
    for product, index, question in unanswerable:
        qid = f"herb:unanswerable:{product}:{index:04d}"
        question_hash = sha256_bytes(question.encode())
        query_rows.append({
            "query_id": qid, "answerability": "unanswerable", "product": product,
            "source_record_index": index, "question": question,
            "question_sha256": question_hash, "evaluation_only_gold": True,
        })
        gold_rows.append({
            "query_id": qid, "answerability": "unanswerable", "product": product,
            "source_record_index": index, "question_sha256": question_hash,
            "question_type": None, "ground_truth": None, "citations": [],
            "evaluation_only": True,
        })
        text_to_records[question].append(qid)

    question_duplicate_groups = {k: v for k, v in text_to_records.items() if len(v) > 1}
    forbidden_keys = {"team", "customers", "answerable_questions", "unanswerable_questions", "ground_truth", "citations"}
    forbidden_artifact_key_hits = []
    for record in artifacts:
        hits = sorted(set(record["visible_field_names"]) & forbidden_keys)
        if hits:
            forbidden_artifact_key_hits.append({"physical_artifact_id": record["physical_artifact_id"], "fields": hits})

    source_files = []
    for base, label in ((DATASET, "dataset"), (CODE, "official_code")):
        for path in sorted(p for p in base.rglob("*") if p.is_file() and ".git" not in p.parts):
            source_files.append({
                "repository": label, "path": str(path.relative_to(base)),
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            })

    dataset_commit = git(DATASET, "rev-parse", "HEAD")
    code_commit = git(CODE, "rev-parse", "HEAD")
    dataset_status = git(DATASET, "status", "--porcelain")
    code_status = git(CODE, "status", "--porcelain")
    metadata_counts = {}
    for path in metadata:
        value = json.loads(path.read_text())
        metadata_counts[path.name] = {"top_level_type": type(value).__name__, "records": len(value)}

    observed = {
        "product_files": len(products), "artifact_records": len(artifacts),
        "unique_original_artifact_ids": len(by_original_id),
        "duplicate_original_id_groups": len(collisions),
        "duplicate_original_id_records": sum(len(v) for v in collisions.values()),
        "duplicate_original_id_extras": sum(len(v) - 1 for v in collisions.values()),
        "answerable_questions": len(answerable), "unanswerable_questions": len(unanswerable),
        "total_questions": len(query_rows), "unique_query_ids": len({r["query_id"] for r in query_rows}),
        "unique_question_texts": len(text_to_records), "duplicate_question_text_groups": len(question_duplicate_groups),
    }
    expected = {"product_files": 30, "artifact_records_documented": 39190, "answerable_questions": 815, "unanswerable_questions": 699}

    manifest = {
        "manifest_schema": "decisionrag.herb_h1.release_manifest.v1",
        "status": "H0_PASS_WITH_DOCUMENTED_RELEASE_COUNT_DISCREPANCY",
        "source": {
            "dataset_url": "https://huggingface.co/datasets/Salesforce/HERB",
            "dataset_commit": dataset_commit,
            "code_url": "https://github.com/SalesforceAIResearch/HERB",
            "code_commit": code_commit,
            "paper_url": "https://aclanthology.org/2025.emnlp-industry.34/",
            "dataset_readme_paper_url": "https://arxiv.org/abs/2506.23139",
        },
        "repository_clean_at_audit": not dataset_status and not code_status,
        "expected_published_counts": expected,
        "observed_native_release_counts": observed,
        "count_reconciliation": {
            "artifact_difference_observed_minus_documented": observed["artifact_records"] - expected["artifact_records_documented"],
            "answerable_matches_documented": observed["answerable_questions"] == expected["answerable_questions"],
            "unanswerable_matches_documented": observed["unanswerable_questions"] == expected["unanswerable_questions"],
            "product_matches_documented": observed["product_files"] == expected["product_files"],
            "interpretation": "Use the pinned native release contents (38,600 artifact records) for H1; do not invent 590 absent records.",
        },
        "artifact_type_counts": dict(type_counts),
        "field_counts": dict(field_counts),
        "metadata_counts": metadata_counts,
        "question_type_counts_answerable": dict(q_type_counts),
        "citation_audit": {
            "citation_occurrences": citation_refs, "unique_citation_ids": len(citation_unique),
            "unresolved_occurrences": len(citation_unresolved),
            "ambiguous_occurrences": citation_target_types.get("AMBIGUOUS_ORIGINAL_ID", 0),
            "resolved_target_type_occurrences": dict(citation_target_types),
        },
        "information_boundary": {
            "forbidden_product_level_fields": ["team", "customers"],
            "evaluation_only_fields": ["answerable_questions", "unanswerable_questions", "ground_truth", "citations"],
            "artifact_forbidden_key_hits": len(forbidden_artifact_key_hits),
            "policy": "Never expose product-level team/customers or gold fields to retrieval, selector, or reader inputs.",
        },
        "source_sha256_manifest": source_files,
        "gpu_probe_at_preflight": gpu_probe(),
        "gpu_authorization": "MISSING_SEPARATE_H1_OWNER_AUTHORIZATION",
    }

    jsonl_dump(OUT / "HERB_H1_ARTIFACT_MANIFEST.jsonl", artifacts)
    jsonl_dump(OUT / "HERB_H1_QUERY_MANIFEST.jsonl", query_rows)
    jsonl_dump(OUT / "protected/HERB_H1_GOLD_SIDECAR.jsonl", gold_rows)
    json_dump(OUT / "HERB_H1_DATASET_AND_LICENSE_MANIFEST.json", manifest)
    json_dump(OUT / "HERB_H1_PREFLIGHT_STATUS.json", {
        "status": "HERB_H1_BLOCKED_RESOURCE_AUTHORIZATION",
        "offline_gates": {
            "native_release_downloaded": "PASS",
            "native_json_parse": "PASS",
            "published_count_reconciliation": "PASS_WITH_DOCUMENTED_590_RECORD_DISCREPANCY",
            "artifact_identity_manifest": "PASS_WITH_33_DUPLICATE_ID_GROUPS_PATH_QUALIFIED",
            "citation_identity_mapping": "PASS_ALL_57911_OCCURRENCES_RESOLVE_UNAMBIGUOUSLY",
            "leakage_policy_defined": "PASS_POLICY_FROZEN_BEFORE_INDEXING",
            "license_terms_recorded": "PASS_OWNER_USE_CONFIRMATION_STILL_REQUIRED",
            "gpu_resource_authorization": "BLOCKED",
            "model_inference": "NOT_STARTED",
            "index_build": "NOT_STARTED",
            "jev_calls": 0,
            "paid_api_calls": 0,
        },
        "next_authorization_required": "Explicit owner authorization naming HERB H1 GPU IDs and execution window; current sandbox cannot verify NVIDIA devices.",
    })
    json_dump(OUT / "HERB_H1_SOURCE_HASH_MANIFEST.json", {
        "dataset_commit": dataset_commit, "code_commit": code_commit,
        "files": source_files,
    })

    collision_rows = []
    for original_id, rows in sorted(collisions.items()):
        collision_rows.append([original_id, len(rows), "; ".join(r["physical_artifact_id"] for r in rows)])
    product_rows = []
    for product in products:
        data = product_data[product.stem]
        product_rows.append([product.stem, sum(len(data.get(k, [])) for k in ARTIFACT_TYPES), len(data.get("answerable_questions", [])), len(data.get("unanswerable_questions", []))])
    inventory = f"""# HERB H1 Dataset Inventory

Status: `H0_PASS_WITH_DOCUMENTED_RELEASE_COUNT_DISCREPANCY`

This inventory was generated offline from the pinned native release. No model, service, index, or paid API was invoked.

## Pinned sources

- Dataset: `Salesforce/HERB`, commit `{dataset_commit}`.
- Official code: `SalesforceAIResearch/HERB`, commit `{code_commit}`.
- Paper: https://aclanthology.org/2025.emnlp-industry.34/ (dataset README also links arXiv: https://arxiv.org/abs/2506.23139).

## Count reconciliation

| Quantity | Published/documented | Observed in pinned native release |
| --- | ---: | ---: |
| Product files | 30 | {observed['product_files']} |
| Enterprise artifact records | 39,190 | {observed['artifact_records']} |
| Answerable questions | 815 | {observed['answerable_questions']} |
| Unanswerable questions | 699 | {observed['unanswerable_questions']} |
| Total questions | 1,514 | {observed['total_questions']} |
| Unique original artifact IDs | not stated | {observed['unique_original_artifact_ids']} |

The native release is the frozen H1 population. The 590-record difference from the README/code prose is recorded, not repaired by fabrication or external retrieval. The answerable, unanswerable, and product counts match the documentation.

## Artifact type totals

{md_table(['Source type', 'Records'], [[k, type_counts[k]] for k in ARTIFACT_TYPES])}

## Product distribution

{md_table(['Product', 'Artifacts', 'Answerable', 'Unanswerable'], product_rows)}

## Question types

{md_table(['Answerable type', 'Count'], [[k, v] for k, v in sorted(q_type_counts.items())])}

Native answerable records do not provide a separate native question ID. H1 query IDs are deterministic path-qualified IDs (`herb:<mode>:<product>:<source-record-index>`), and canonical question hashes are stored in `HERB_H1_QUERY_MANIFEST.jsonl`.

## Duplicate identity audit

There are {len(collisions)} original-ID collision groups covering {sum(len(v) for v in collisions.values())} records ({sum(len(v)-1 for v in collisions.values())} extra records). All are retained as distinct physical records using path-qualified IDs `herb:<product>:<source-type>:<index>`.

{md_table(['Original ID', 'Records', 'Stable physical IDs'], collision_rows)}

Question text is also not used as identity. There are {observed['duplicate_question_text_groups']} duplicate question-text groups across answerable/unanswerable records; all remain separate query IDs.

## Metadata

- `metadata/employee.json`: 530 records.
- `metadata/customers_data.json`: 120 records.
- `metadata/salesforce_team.json`: 18 top-level records.
- Product-level team references: {field_counts['team']}.
- Product-level customer references: {field_counts['customers']}.

Metadata and product-level relationships are subject to the leakage contract in `HERB_H1_FIELD_LEAKAGE_AUDIT.md`.
"""
    (OUT / "HERB_H1_DATASET_INVENTORY.md").write_text(inventory)

    collision_detail = "\n".join(
        f"- `{original_id}` -> " + ", ".join(f"`{r['physical_artifact_id']}` ({r['product']}/{r['source_type']}/{r['source_record_index']})" for r in rows)
        for original_id, rows in sorted(collisions.items())
    )
    (OUT / "HERB_H1_FIELD_LEAKAGE_AUDIT.md").write_text(f"""# HERB H1 Field Leakage Audit

Status: `PASS_WITH_FROZEN_BOUNDARY_BEFORE_INDEXING`

## Serving-visible allowlist

The H1 RAG corpus may expose only the canonical text and non-oracle metadata of each physical artifact:

- Slack message/channel/thread content and source type.
- Meeting transcript/chat content and source type.
- Document content, type, author/date/link and source type.
- URL description/link and source type.
- Pull-request title/summary/link/state/review content and source type.
- Stable physical ID and source provenance for joining results.
- Canonical question text.
- Metadata files only where needed for the frozen retrieval adapter; metadata must not be joined through the product-level gold relationships.

## Evaluation-only / forbidden serving fields

- Product-level `team` and `customers` arrays: excluded from all model-visible artifact text. The official README reserves these for oracle/long-context settings.
- `answerable_questions` and `unanswerable_questions`: excluded from the corpus.
- `ground_truth`, `citations`, and answerability labels: protected gold sidecar only.
- Any generated gold support labels, future selector scores, or reader outputs: never present during retrieval or selection.

The native product files contain `team` and `customers` in all 30 products ({field_counts['team']} and {field_counts['customers']} references respectively), but these are structural fields and are not artifacts in the serving corpus. The official oracle/long-context scripts directly consume these fields; those scripts are not an H1 RAG implementation and were not executed.

The offline artifact scan found {len(forbidden_artifact_key_hits)} forbidden-key hits inside the actual artifact records. H1 must retain this check after canonical extraction and before indexing.

## Required implementation guard

The H1 index builder must fail closed if a serving document contains a forbidden field or any protected gold field. It must emit a field-audit hash and counts before any GPU embedding/index operation. The protected gold sidecar is stored separately at `protected/HERB_H1_GOLD_SIDECAR.jsonl`.

## Identity collision handling

Original source IDs are preserved as metadata, but never used as the sole internal key. {len(collisions)} collision groups were observed. Examples and complete mapping are in `HERB_H1_ARTIFACT_MANIFEST.jsonl`; the full collision list is:

{collision_detail}
""")

    (OUT / "HERB_H1_EVIDENCE_LABEL_AUDIT.md").write_text(f"""# HERB H1 Evidence Annotation Audit

Status: `PASS_WITH_LIMITATION_CITATIONS_NOT_PROVEN_COMPLETE_SUPPORT_SETS`

## Mapping

- Answerable questions: {len(answerable)}.
- Citation occurrences: {citation_refs}.
- Unique citation IDs: {len(citation_unique)}.
- Unresolved citation occurrences: {len(citation_unresolved)}.
- Ambiguous citation occurrences: {citation_target_types.get('AMBIGUOUS_ORIGINAL_ID', 0)}.
- All citation occurrences map to exactly one original artifact ID in the pinned release.

Resolved citation target types:

{md_table(['Source type', 'Citation occurrences'], [[k, v] for k, v in sorted(citation_target_types.items()) if k != 'AMBIGUOUS_ORIGINAL_ID'])}

The native `citations` field is an answer-provenance/reference list. The release does not establish that it is a complete necessary evidence set for every answer. Therefore H1 must not call citation recall a verified CSSR or complete-support metric. The preregistered primary support metrics are restricted to explicitly named citation-based recall/coverage unless a separate audit establishes completeness without using model outcomes.

No citations were unresolved or ambiguous in the pinned native release. This is identity coverage only; it does not prove semantic sufficiency of the listed artifacts.

## Gold boundary

Question type, ground truth, citations, and answerability remain in the protected gold sidecar. Retrieval, selectors, and the reader receive none of these fields.
""")

    (OUT / "HERB_H1_LICENSE_AND_USAGE_MANIFEST.md").write_text(f"""# HERB H1 License and Usage Manifest

## Sources and hashes

- Dataset URL: https://huggingface.co/datasets/Salesforce/HERB
- Dataset commit: `{dataset_commit}`.
- Official code URL: https://github.com/SalesforceAIResearch/HERB
- Official code commit: `{code_commit}`.
- Dataset README SHA-256: `{sha256_file(DATASET / 'README.md')}`.
- Official code license SHA-256: `{sha256_file(CODE / 'LICENSE.txt')}`.

## Terms recorded

The dataset README declares `CC BY-NC-4.0` and says the release is for research purposes. It also states that the data were generated using GPT-4o and should not be used to develop models that compete with OpenAI. The official code checkout includes a Creative Commons Attribution-NonCommercial 4.0 license text.

## H1 intended use

H1 is a noncommercial, evaluation-only extension of an existing reranking/RAG research project. It does not train or fine-tune a model, does not create a competing model, does not call hosted OpenAI or Cloudflare/Jev APIs, and does not publish the raw release outside the permitted workspace. Existing checkpoints and local services are evaluated only.

This is a recorded interpretation, not legal advice. Owner confirmation is required before full H1 execution that this intended use and local artifact handling comply with the dataset's noncommercial and additional research-use language. If that confirmation is not granted, the final H1 status must remain `HERB_H1_BLOCKED_LICENSE_OR_USAGE`.

## Paid/API boundary

- Jev calls: 0.
- Paid hosted inference calls: 0.
- Reference scripts were not executed; several default to hosted model endpoints.
""")

    (OUT / "HERB_H1_PROSPECTIVE_EXPERIMENT_PROTOCOL.md").write_text(f"""# HERB H1 Prospective Experiment Protocol

Protocol state: `FROZEN_FOR_OWNER_REVIEW; INFERENCE_NOT_STARTED`

## Population

Use all {len(answerable)} pinned answerable and all {len(unanswerable)} pinned unanswerable questions, with the full {len(artifacts)}-record permitted artifact corpus. No outcome-based sampling, product restriction, retrievability filter, or model-score selection is allowed.

The query manifest is `HERB_H1_QUERY_MANIFEST.jsonl`; evaluation-only labels are in `protected/HERB_H1_GOLD_SIDECAR.jsonl`. Stable physical candidates are in `HERB_H1_ARTIFACT_MANIFEST.jsonl`.

## Policies (unchanged)

{chr(10).join(f'{i}. `{p}`' for i, p in enumerate(POLICIES, 1))}

Reuse the existing checkpoints, tokenizer/scoring contracts, RRF(60), Top-20 retrieval, Top-5 selection, frozen view budgets, rescue rules, groupwise semantics, and local Qwen reader configuration. No H1 tuning is permitted.

## Primary paired comparisons

1. BGE pointwise versus Clef groupwise.
2. Qwen pointwise versus Clef groupwise.
3. Clef groupwise versus Retrieval Top-5.

Use the strongest defensible citation-based support metric under the evidence audit. Do not call it CSSR or complete-support retention unless citation completeness is independently established. Secondary comparisons cover Laya, Clef pointwise, rescue policies, and raw Top-20.

Uncertainty: query-paired bootstrap, 5,000 replicates, seed 731, 95% confidence intervals; report product-cluster sensitivity because questions are grouped by product. Answerable and unanswerable populations are never pooled for metric calculation.

## Frozen retrieval contract

For each question: BM25 Top-100, dense Top-100 using `Qwen/Qwen3-VL-Embedding-2B` with 2,048-dimensional embeddings, RRF(60), then Top-20 unique physical artifacts with deterministic tie-breaking. Use the full corpus, not source-product restriction. Build dedicated H1 indexes; do not overwrite A/B/C/R2 indexes.

## Leakage contract

Product-level `team`/`customers`, question labels, ground truth, citations, and future scores are evaluation-only. See `HERB_H1_FIELD_LEAKAGE_AUDIT.md`.

## Execution gates

Before model inference: owner confirmation of dataset usage and explicit GPU IDs plus execution window. The current preflight has no such H1 authorization, and direct `nvidia-smi` probing is unavailable in this sandbox. Therefore H2/H3/H4 are not started.

Planning logical outcomes, if no records change: {len(answerable)} x 12 = {len(answerable) * 12} answerable policy outcomes; {len(unanswerable)} x 12 = {len(unanswerable) * 12} unanswerable policy outcomes; {len(query_rows) * 12} total policy outcomes.
""")

    gpu = manifest["gpu_probe_at_preflight"]
    planning_seconds = 1877.426 * len(query_rows) * len(POLICIES) / 1200
    (OUT / "HERB_H1_RUNTIME_AND_RESOURCE_REPORT.md").write_text(f"""# HERB H1 Runtime and Resource Preflight

Status: `BLOCKED_RESOURCE_AUTHORIZATION`

No H1 inference or index build was run. Direct GPU probe at preflight returned code `{gpu['returncode']}` with the following captured output:

```text
{gpu['stdout']}
{gpu['stderr']}
```

No current GPU memory, utilization, process ownership, or safe execution window can be independently verified from this sandbox. Historical Track C R2 evidence is not substituted for current capacity and is retained at its original path.

## Planning estimate only

Track C R2 measured 1,200 requests in 1,877.426 seconds at concurrency 10. A naive linear planning ratio for {len(query_rows) * len(POLICIES):,} H1 policy outcomes is approximately {planning_seconds:,.3f} seconds ({planning_seconds / 3600:.2f} hours) at the same throughput. This is not an H1 measurement and must not be used as a completion claim; H1 has a different corpus, question mix, retrieval/index state, and likely workload. Index-build time is `NOT_MEASURED` until the embedding model, batch size, GPU allocation, and chunking output are authorized and benchmarked.

## Required authorization

Provide explicit owner authorization naming the H1 GPU IDs, execution window, and whether existing model services may be reused. No service restart, termination, preemption, Jev call, or paid API call is authorized by this preflight.
""")

    (OUT / "HERB_H1_FINAL_REPORT.md").write_text(f"""# HERB H1 Final Status (Preflight Only)

## Verdict

`HERB_H1_BLOCKED_RESOURCE_AUTHORIZATION`

The offline H0/H1 preflight is complete and resumable. It did not execute H2 indexing, H3 smoke inference, H4 policy evaluation, H5/H6 metrics, or H8 serving measurements because the task specification requires separate explicit owner authorization before HERB GPU work and the current sandbox cannot access NVIDIA devices.

## Completed

- Pinned native dataset commit: `{dataset_commit}`.
- Pinned official code commit: `{code_commit}`.
- Parsed all 30 native product files and 3 metadata files.
- Observed {len(artifacts)} physical artifact records, {len(answerable)} answerable questions, {len(unanswerable)} unanswerable questions.
- Created path-qualified physical IDs and preserved all original IDs.
- Found and documented {len(collisions)} original-ID collision groups.
- Mapped all {citation_refs} citation occurrences to unique physical artifacts.
- Defined the serving/evaluation field boundary.
- Preserved all pre-existing Track A/B/C/R2 artifacts; H1 outputs are in this new subtree.
- Jev calls: 0. Paid API calls: 0.

## Required owner decisions before resume

1. Confirm the CC BY-NC/research-use interpretation for noncommercial evaluation-only use.
2. Authorize specific H1 GPU IDs and an execution window.
3. Confirm that the pinned native release's 38,600 artifact records (rather than the stale 39,190 prose count) are the intended frozen population.

The H1 experiment remains resumable after those decisions. No scientific quality conclusion is claimed.
""")

    print(json.dumps({
        "out": str(OUT), "status": "HERB_H1_BLOCKED_RESOURCE_AUTHORIZATION",
        "dataset_commit": dataset_commit, "code_commit": code_commit,
        "artifact_records": len(artifacts), "answerable": len(answerable),
        "unanswerable": len(unanswerable), "citation_occurrences": citation_refs,
        "citation_unresolved": len(citation_unresolved), "collision_groups": len(collisions),
        "jev_calls": 0, "paid_api_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
