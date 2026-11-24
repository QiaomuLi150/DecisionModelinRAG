# E8 source authentication and read isolation

E7 gate authenticated as `E7_VALIDATED_EVIDENCE_READY`. The exact E7-B cohort is used without resampling: 40 SciFact and 40 GaRAGe rows, 80 unique query keys. The cohort validator found **0** pre-inference problems.

Inference inputs are constructed from the label-blind E7 cohort only. Relevance annotations, reference answers, protected/reserved files, E5-F/E6-F/E3-F material and Jev endpoints are not opened by the inference runner. The new rank field is tied to canonical candidate identity and remains unchanged under reversal.

Source hashes and the pre-outcome freeze are in `E8_PREINFERENCE_PROTOCOL.json` and `E8_FREEZE_MANIFEST.json`. Historical E7/E6 files are not modified.

## Validation findings

- PASS: candidate IDs, full text, question hashes, canonical ordering, uniqueness, and label-blindness validated.
