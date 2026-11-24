# E3-B Authenticity and Independent Replay

The B2 replay independently loads raw E2-M1 score vectors and raw quartet assignments, validates 20-ID alignment and five groups of four, then recomputes unrestricted and one-per-quartet selections. B3 independently loads raw BGE score vectors and computes TF-IDF similarities and selections without qrels in the selector. All gold use occurs after selection for offline metrics. B1 has no fabricated output: blocked rows carry `physical_forward_count=0` and an explicit failure reason.
