# v3.6 follow-up deviations

- The corrected Jina rerun remains preserved, but its exact-list agreement with the earlier frozen trace is 1126/1352 (83.284%) and set agreement is 1276/1352 (94.379%), below the prespecified 99% gate; Jina-derived claims remain unverified.
- N2 session C includes required Jina/PPLX anchors and local CrossEncoder controls. Drex-v2 was run in a separate GPU-4 HTTP server session after the initial undersized-batch 400 errors were preserved; it has no same-session anchors and is reported as partial.
- N2 session E contains a corrected native Winnow run and same-session Jina/PPLX anchors, but its pre-inference protocol receipt was not written before the anchor phase. The branch is therefore descriptive/partial and is not promoted to a primary matched N2 gate. The initial resume pass skipped Winnow because of a policy-key bug before any Winnow request was sent; the corrected run executed the intended 1080 Winnow requests.
- N3 transfer is not run because the available v2.1 confirmation slates are hybrid RRF and no exact BM25-only source artifact is available.
- Device indices, UUIDs and process details are retained only in raw provenance; exports use slots and are sanitized before packaging.
