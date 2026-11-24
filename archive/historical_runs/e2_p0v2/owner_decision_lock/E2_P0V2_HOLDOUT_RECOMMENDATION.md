# E2-P0v2 Holdout Recommendation

**Status:** `PROPOSED_CANDIDATE_BLOCKED_PENDING_VERIFICATION`

## Preferred candidate

Use the local 2Wiki test release as the first candidate for an untouched holdout, because it is the only locally present test-named source among the three multi-hop sources that was not parsed in this phase. The metadata-only inventory records its physical path, size, and SHA-256 in the machine-readable provenance. This is a candidate, not a valid sealed holdout.

Required checks before acceptance:

1. Verify official release identity, revision, license, and intended split.
2. Audit query IDs and normalized question text against Track A, Track B, HERB, Track C, CAL, DEV, debug, and all M0 populations.
3. Audit source-document IDs and normalized source content for overlap.
4. Verify annotation completeness and evidence identity mapping without exposing labels to model selection.
5. Establish a protected gold-sidecar access boundary, deterministic sampling seed, exact `n`, and immutable hash lock.
6. Run the independence audit before any model outcome is inspected.

## Fallback

Acquire an official, separately versioned HotpotQA, MuSiQue, or 2Wiki evaluation release with a documented untouched split and protected annotations. The local HotpotQA and MuSiQue files are historical development/train sources in this workspace and are not automatically independent. The fallback remains `EXTERNAL_VERIFICATION_REQUIRED`; no download is authorized by this phase.

## Current evidence and limitations

The prior ID-only audit found zero source-record overlap between historical CAL/prior-test records, debug fixtures, and the historical `NEW_TEST_IDS` manifest for HotpotQA, MuSiQue, and 2Wiki. That does not prove independence: source-document reuse, near-duplicate questions, release identity, and annotation provenance remain unverified. No fresh gold labels were opened.

No existing Track A, Track B, HERB, Track C/R2, or M0 population may be relabeled as confirmatory.
