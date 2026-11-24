# Tier 2 inference status

GPU inference was not run during packaging. The original runners depend on model weights, tokenizer assets, runtime packages, and local launch configuration that are intentionally not shipped. The package therefore does not provide a turnkey Tier 2 command or claim that the sanitized historical analysis scripts can run the native model stack.

Before any future inference, resolve the full Jina v3.5 revision in `MODEL_PINS.md`, obtain the exact approved data revision, install the model vendor's pinned native interface, and use the immutable protocols under `protocols/`. The paper's PPLX adapter is its official `DecisionModel.predict` interface with calibration temperature and noncausal attention—not plain causal-LM generation. PPLX single-call uses one score vector/forward; two-stage uses three forwards. These are contract notes, not executable commands.

No model weights, tokenizer files, process lists, credentials, or GPU UUIDs are included.

To download a model snapshot manually, use `python3 scripts/download_model.py --model pplx --output-dir <directory-outside-this-repository>`. This optional command requires `huggingface_hub`; it was not run for this package. Models without a complete historical revision require an explicit full revision plus `--accept-unverified-revision` and must not be described as exact historical reproductions.
