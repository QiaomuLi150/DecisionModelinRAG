# E6-3A Serving Correction Report

The correction is frozen as `E6_3A_SERVING_CORRECTION_SPEC.json` before any corrected model request. It changes only the SciFact candidate-view reconstruction from empty missing fields to the authenticated E6-2 `title + newline + text` source view. It retains the exact historical 100-query sample, candidate order, revisions, model adapters, label-blind policy and two-repetition design. GaRAGe inputs are reused unchanged.

The historical E6-3 raw files remain authoritative evidence of the defect and are not edited. Corrected rows are stored under `corrected_raw/` and have separate hashes. No quality outcome, relevance label, reserved record, reader answer or Jev/paid API is accessed by the correction runner.
