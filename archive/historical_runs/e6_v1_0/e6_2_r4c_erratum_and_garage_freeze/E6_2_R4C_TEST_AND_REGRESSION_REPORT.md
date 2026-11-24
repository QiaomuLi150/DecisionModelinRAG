# E6-2 R4C tests and regression

Focused offline implementation checks: **13/13 passed**; the committed pytest regression module adds **8/8 passed** checks. They cover corrected Top-20 membership, 254/277 reproduction, primary reproduction, K15 eligibility/alignment, component construction, deterministic non-overlap, outcome-blind allocation, label isolation, reservation boundary and zero inference/API calls.

The authoritative shared-environment full regression suite completed **245/245 passed** (the prior 237-test baseline plus the 8 new R4C tests). This stage performed no GPU/model/reader inference, no retrieval, no embedding generation, no Jev call and no paid API call.
