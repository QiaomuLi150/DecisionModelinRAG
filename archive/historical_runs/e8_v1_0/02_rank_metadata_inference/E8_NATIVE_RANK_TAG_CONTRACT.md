# E8 native rank-tag contract

The only intervention is the fixed identity-tied field `original_retrieval_rank=NNN`, serialized as a prefix followed by a newline and the authenticated full candidate text. Candidate ranks are 1-based from the canonical E7 slate and are unchanged under reversal. C uses canonical presentation order; D reverses presentation order while retaining each candidate's original tag. No labels, qrels, reference answers, model scores or outcomes enter the input.

The intervention is diagnostic and does not represent the unmodified native policy. Full text is required; context overflows remain explicit failures. Output validity requires a native score/readout aligned to presented candidate positions, five unique canonical IDs, membership in the presented candidate set, and preserved rank/ID mapping. Raw output is frozen in `E8_RANK_ANNOTATED_RAW_OUTPUTS.jsonl` before any offline label join.
