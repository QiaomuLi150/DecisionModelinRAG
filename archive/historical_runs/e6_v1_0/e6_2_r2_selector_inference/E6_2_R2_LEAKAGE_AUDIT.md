# E6-2 R2 Leakage and Reservation Audit

Status: PASS for CPU preflight; selector inference not started.

The preflight read only the frozen R1 slates, public SciFact corpus metadata/content needed for candidate-hash verification, prior model registries, and reservation metadata. It did not read qrels, relevance labels, reference answers, reader outputs, RESERVED question text, or R3 outputs. No model weights were loaded and no selector request was made.

Fresh device-specific GPU authorization is absent. The earlier general statement that idle GPUs may be used is not reused as R2 device authorization because the R2 SDD explicitly requires current GPU UUID authorization.
