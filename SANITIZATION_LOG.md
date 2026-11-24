# Sanitization and privacy checks

All selected source files are staged as copies. The experiment repository remains unchanged. Text and structured files are scanned and sanitized in the release copy; source trace projections drop literal benchmark content, process/host fields, personal contact fields, and GPU UUIDs. The field names removed are listed below; removed values are not retained here.

Fields removed from structured records when present: query/passage/document text; raw prompt/request/response text; absolute path, root, host, username, email, process-list and process-command fields; GPU UUID/device UUID and physical-device identity; API key/token/credential fields; label-derived fields from raw serving traces.

Additional release-only cleanup: path-bearing or private-marker values in eligible ID fields are replaced with deterministic SHA-256 pseudonyms; benchmark text fields such as TREC segments and local document URLs are removed. GPU snapshot/process fields and process-list records are omitted from exported data. The per-file source and sanitized hashes remain in `archive/HISTORICAL_FILE_CROSSWALK.csv`.

Additional identifier transformation: in `data/scores/e2_n_top5_ids.jsonl.gz` and `raw_traces/e2_n_native_scores.jsonl.gz`, path-like, private-marker, or credential-shaped values occurring in `candidate_ids` or `selected_ids` were replaced with deterministic SHA-256 pseudonyms. This retains stable identity across released projections; the original-to-pseudonym mapping is not included. The transformation changed 58 unique ID values in the Top-5 projection (1,014 occurrences) and 58 unique values in the raw score projection (1,820 occurrences). No result scores, candidate order, selection membership, or counts were changed.

Text substitutions: absolute paths, home directories, local mount paths, host/user strings, internal IPs/URLs, email addresses, and credential-shaped strings are replaced by `<redacted>` or removed. Public dataset/model/arXiv URLs are retained where necessary to point to official sources.

Additional owner-approved pre-publication cleanup was applied to this release copy after the initial packaging scan: 28,621 path-pattern substitutions across 32 files, including six gzip payloads. No username, credential, or GPU UUID substitutions were recorded in that pass. The source experiment workspace was not edited. Release-side checksum, trace-manifest, source-crosswalk, historical-crosswalk, and protocol hash-table values were refreshed. Because this is substring-level sanitation, affected text-bearing records are not guaranteed to preserve exact input bytes; their use for byte-identical inference replay is limited.

Final hygiene status and the exact scope of compressed-aware checks are recorded in `SANITIZATION_SCAN.txt`. The staged copies were scanned without printing matched values. The bundled `verify/scan_release.py` is available for owner-side rechecking; it recursively scans gzip, ZIP and tar members.

Raw source traces containing third-party query/document text are not copied; score-only projections and source hashes are released when safe. This is a data-minimization decision, not a change to reported metrics.
