# Public-release sanitation report

Date: 2026-11-24

## Scope and result

The release copy was scanned and sanitized before its Git history rewrite. The recursive content scan covered 2,922 files and decompressed gzip payloads; it passed with no private absolute-path markers, private username markers, credential-shaped tokens, GPU UUIDs, or captured process listings. The report does not reproduce any matched values.

The sanitizer changed path-like/private-marker substrings in 32 release files, including six gzip payloads. Its recorded count was 28,621 path-pattern substitutions. No username, credential, or GPU-UUID substitutions were recorded. Per-file release hashes, trace projection hashes, and crosswalk release hashes were refreshed; original source hashes remain provenance metadata and source payloads are not distributed.

No score, ranking metric, label, or published aggregate was intentionally edited. The substitutions operate on matching strings wherever they occur, so a redacted text-bearing record may not be byte-identical to the private inference input. Such records support inspection and selected analyses, but not exact-input replay. See `MISSING.md` and `REPRODUCIBILITY.md` for the broader reproduction limits.

## History and visibility gate

The repository was private at the start of this cleanup. The owner authorized rewriting the already-pushed history. A full tree-by-tree rewrite stopped when an older snapshot contained a marker inside a binary artifact; the published history was replaced by one clean root commit from the verified current release tree. The release tree is retained, while the former commit graph is not. A private pre-cleanup backup retains the prior history outside the repository. The local reachable history and release-tree scans passed, but GitHub's authenticated commit API still returns the three pre-redaction commit IDs after the force update. Visibility remains private. GitHub's sensitive-data guidance calls for Support assistance to purge cached views and references; no support request has been sent. Do not make the repository public until those old commits are no longer served.

## Verification commands

```sh
python3 verify/scan_release.py
python3 verify/audit_git_history.py
python3 verify/verify_hashes.py
python3 verify/run_integrity_tests.py
python3 verify/compile_scripts.py
```

The independent historical test suite is not fully reproducible from the package: the recorded run collected 297 tests, with 70 passed, 225 failed, and 2 skipped because required historical inputs and result artifacts are absent. This package therefore supports bounded integrity checking and selected offline analyses, not complete end-to-end reproduction.
