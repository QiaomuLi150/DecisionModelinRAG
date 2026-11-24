# E3-B SetR Feasibility Audit

Verdict: `SETR_NOT_REPRODUCIBLE`

The official repository is [LGAI-Research/SetR](https://github.com/LGAI-Research/SetR), licensed BSD-3-Clause, and the paper is [ACL 2025, DOI 10.18653/v1/2025.acl-long.861](https://aclanthology.org/2025.acl-long.861/). The official README provides training-data generation, training, and response-generation instructions; its evaluation step is explicitly `TBD`. The repository page exposes no release/package/checkpoint artifact, and the permitted local HF cache has no SetR checkpoint (`local_cache_matches=[]`).

The documented path depends on generating responses/training a Llama-based selector and does not provide a verified pretrained Top-20-to-Top-5 inference contract for this project's frozen query/candidate representation. No repository was cloned, no weights were downloaded, no SetR inference was run, and no training was performed. Therefore the E3-B set-aware controls are explicitly `BGE_MMR` and `BGE_COVERAGE` TF-IDF heuristics, never SetR or a learned SetR equivalent.
