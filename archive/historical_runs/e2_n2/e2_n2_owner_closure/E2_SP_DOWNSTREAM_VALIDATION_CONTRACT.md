# E2-P Downstream Validation Contract

Status: `READY_FOR_SEPARATE_AUTHORIZATION`; no reader run performed.

Hold the reader checkpoint, prompt, decoding, evidence layout, context budget, and citation schema fixed. Compare rerankers on a prespecified CAL/DEV cohort, retaining the same retrieved candidate slate. Report independently reviewed factual correctness, evidence/citation completeness, abstention, reader latency, tokens, and modeled cost. Reference-answer EM/F1 are secondary proxies only; CSSR, answer nonemptiness, and model-generated labels are not factual correctness. Do not access a fresh confirmatory holdout in this phase.
