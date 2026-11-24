# E2-P0v2 Authenticity and Telemetry Specification

## Before inference

Fail closed on empty canonical question, missing/duplicate/out-of-slate candidate IDs, unverified adapter, unapproved model revision, score-contract mismatch, gold access, missing config hash, or missing GPU reservation. Validate candidate and query manifests, prompt/schema hashes, model revision/checkpoint hashes, CAL/DEV/TEST separation and Jev denial.

## Per call and per result

Capture raw typed response bytes before normalization, model-call ID, canonical question hash, ordered candidate IDs, actual model-visible input hash, selection and score-contract version, raw readout, candidate alignment, tokenizer/truncation/overflow, model batch size, physical forward/pass counters where instrumentable, start/end timestamps, queue wait, GPU allocation/utilization/memory, cache key and exact-input cache provenance, retries, errors and fallback.

## During/after serving

Validate one score per candidate where required, stable ties, no hidden filtering, five unique selected IDs, full request boundary, cache-disabled behavior, physical-work accounting and immutable output hashes. Any unexplained cache reuse or missing physical evidence produces an explicit invalid/unsupported outcome.

CPU mocks must cover empty input, duplicate IDs, score alignment, tie-breaks, truncation, cache provenance, retry/error accounting and gold-leakage denial.
