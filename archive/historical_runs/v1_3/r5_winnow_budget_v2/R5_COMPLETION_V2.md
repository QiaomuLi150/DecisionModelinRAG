# R5 completion status

Gate: `R5_BLOCKED_DEDICATED_GPU_UNAVAILABLE`

The exact Winnow runtime blocker from the earlier stage was resolved at the build/provenance level. The pinned native server compiles and its unit test passes. The actual R5 benchmark remains unexecuted because the frozen protocol requires an isolated dedicated A100, while the current host snapshot shows a resident process on every GPU. Sharing GPU4 with resident VLLM services or GPU3 with an unattributed process would invalidate the requested matched serving measurement.

Completed: exact source checkout, locked llama.cpp checkout, CUDA build, binary hash, model/tokenizer hashes, native endpoint contract audit, and resource-gate record.

Not executed: Winnow or Jina R5 inference, selection outputs, latency, peak memory, HTTP-error stderr capture, or quality scoring. No R5 result has been fabricated or substituted with a non-equivalent runtime.

The prior `results/v1_3/r5_winnow_budget/R5_BLOCKER.json` is preserved unchanged; this version supersedes only its diagnosis that the exact binary was unavailable.
