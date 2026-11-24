# Reproducibility audit

## Verdict

**Partial reproducibility; the whole DecisionRAG project is not reproducible from this public repository alone.** A clean checkout can verify the released file hashes and run the bounded release-integrity suite. It cannot independently rerun all historical experiments or regenerate every manuscript result because important source datasets, model weights/tokenizers, raw inference records, runtime services, and some stage result bundles are not included.

This is an evidence/package limitation, not a claim that the historical findings are false. Missing claims and inputs remain listed in [`MISSING.md`](MISSING.md). No missing results were reconstructed or substituted during this audit.

## Checks available from this checkout

Run from the repository root with Python 3.12:

```sh
python3 verify/verify_hashes.py
python3 verify/run_integrity_tests.py
python3 verify/compile_scripts.py
python3 verify/scan_release.py
python3 verify/audit_git_history.py
```

The hash verifier checks the package checksum list, 73 source-to-release crosswalk entries, 29 trace projections and the frozen CQADupStack protocol/cohort hashes. It ignores generated Python/pytest caches so running the checks does not make the release appear modified. The integrity script is a small release check, not the complete historical regression suite. Syntax compilation checks packaged Python scripts in memory without generating `.pyc` files; it does not validate model interfaces or reproduce model outputs. The history audit checks reachable Git blobs for private path markers, credential patterns, GPU UUIDs and process-dump markers; it reports category counts only if it finds a match.

This audit also repaired sanitizer-induced syntax damage in 36 packaged historical scripts and restored one missing offline analysis module from the available project source. These are source-package repairs only; frozen result files were not changed. All packaged scripts compile, but that does not make omitted stage inputs or model runtimes available.

The historical tests can be explored with:

```sh
python3 -m pytest -q code/tests
```

Audit run in this environment: **297 collected; 70 passed, 225 failed, 2 skipped**. The suite is not green from the release. For example, its first failure requests `code/results/non_jev/v2/e2_m0/E2_M0_OWNER_GATE.json`, which is not shipped. Other tests expect historical per-stage outputs, source data, and runtime/model files. The failures are left visible; tests and checks were not weakened to obtain a pass. Review the full suite against the inputs listed in `MISSING.md` before interpreting individual failures as code defects.

## What can and cannot be reproduced

| Area | Status from this repository | Boundary |
|---|---|---|
| Release file integrity | Reproducible | File SHA-256 list, source crosswalk, trace projections, and selected frozen-protocol hashes are checked by `verify/verify_hashes.py`. |
| Bounded release integrity tests | Reproducible | `verify/run_integrity_tests.py`; these do not assert every historical experiment. |
| Python syntax | Reproducible | Packaged scripts compile under the checked Python 3.12 runtime; model execution is not exercised. |
| Selected tabular/offline analyses | Partially reproducible | Some source code and derived tables are included. Reproduction depends on the specific inputs each analysis needs; the package does not provide every historical raw trace or benchmark label. |
| Full historical regression suite | Not reproducible as a passing suite | 225 of 297 tests failed in the audit, chiefly where test contracts require stage artifacts omitted from this release; the first missing artifact is documented above. |
| Native model inference | Not turnkey/reproducible | Model weights, tokenizers, vendor runtime packages, service launch settings, and GPU execution environment are not distributed. |
| Every manuscript result and figure | Not fully independently reproducible | The complete historical model-by-benchmark raw score matrix, some serving traces, exact pins, source dataset revisions, and figure-generation inputs are unavailable. See `MISSING.md` and `verify/expected_numbers.json`. |
| Paper source build | Not verified | The supplied TeX source and PDF are included, but this audit did not build the PDF or establish byte identity with the current arXiv source archive. |

## Inputs needed for stronger reproduction

1. Obtain benchmark corpora, qrels, and source documents from their official releases, after reviewing each license; pin exact revisions and record downloaded-file hashes.
2. Recover the missing frozen candidate pools, annotations, stage manifests, and raw traces identified in `MISSING.md` from their authorized owners. Do not reconstruct absent runs from paper summaries.
3. Obtain the exact model revisions, tokenizers, official native adapters, runtime dependency lockfiles, and launch configurations. The historical Jina full revision is not fully authenticated in the packaged evidence.
4. Provide an isolated compatible GPU and the required serving services for any inference reproduction; no inference is run by the package-integrity commands.
5. Reconcile remaining manuscript values against the original source records before treating them as independently reproduced.

The large source artifacts listed in `RELEASE_ASSETS_TODO.md` were excluded because they contain benchmark/model-input material or have unresolved redistribution rights. No release download is claimed for them. The repository's data-redistribution policy is in [`LICENSE-DATA.md`](LICENSE-DATA.md).

## Public-sharing gate

The current release tree passed the recursive privacy scanner across 2,921 files, including compressed payloads. The sanitation changed path-like/private-marker strings in 32 release files, including six gzip traces; the recorded run counted 28,621 path-pattern substitutions. No credential-shaped string, username marker, or GPU UUID was substituted in that pass. The sanitized release hashes and trace/crosswalk manifests were refreshed. This is release-copy hygiene: exact serialized inputs in affected text-bearing records may no longer be byte-identical to the original private files, so those records must not be presented as exact replay inputs. The sanitation did not edit score values or metric fields.

The owner-authorized history rewrite replaces the old commit graph with one clean root commit built from the sanitized release tree. This keeps all release files but does not preserve the old Git commit graph. A private pre-cleanup backup retains the prior history outside the repository. The rewritten tree and local reachable history passed the scanner, but GitHub's authenticated commit API still returns the three pre-redaction commit IDs after the force update. The repository therefore remains private. GitHub's sensitive-data guidance calls for Support assistance to purge cached views and references; no support request has been sent. The final visibility and history-scan status are recorded in `REPRODUCIBILITY_STATUS.json`.
