# E3-C Source and Model Audit

Historical source population: 4,267 unique logical queries: Track A 3,767 and Track B 500. Frozen 20-candidate identities and original text-view hashes were independently validated. Gold is loaded only by the post-inference analysis path; neither native adapter receives gold qrels, reference answers or CSSR fields.

Liquid: `LiquidAI/d1-3B` at `051bcc464b01b9f92942b364d9586b0ef5912432`, local snapshot `<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432`, LFM-1.0 license. Official remote-code files `prompt.py`, `runner.py`, `modeling_d1.py`, and `api.py` are present. Zerank: `zeroentropy/zerank-2-reranker` at `5eae30d5ee3c6b2df2ef6d723bde45172d761c4c`, local snapshot `<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--zeroentropy--zerank-2-reranker/snapshots/5eae30d5ee3c6b2df2ef6d723bde45172d761c4c`, Apache-2.0 license. Only these two new checkpoints were downloaded. Jev/paid APIs, holdout outcomes, training and multimodal inputs were not accessed.

Source hashes and complete model file inventories are in `E3_C_FROZEN_EXPLORATORY_PROTOCOL.json`.
