# R5 exact-runtime build report

Status: `RUNTIME_READY_EXECUTION_BLOCKED_BY_DEDICATED_GPU_GATE`

The previously missing exact runtime was built in an isolated temporary worktree. The build used the pinned Winnow source commit `6c2b3c04e248a319f2cb43832628eba03e55fe38`, the locked llama.cpp commit `911f6cdc8ab8a530b2bee09ee61471a6f3178eeb`, CUDA architecture 80, and the exact build command recorded in `PROTOCOL_V2.json`.

The resulting binary is `<REDACTED_PATH>` with SHA-256 `2f729cd3f59b34dc5b3a9a7cd0c5096e59075fe8dcd246529e9b34753f273993`. The build completed at 100%, and the exact runtime unit test passed: `1/1` tests passed. The binary reports version `0.4.1-dev`, build `11036`, commit `911f6cdc8`.

The local Q8 model and tokenizer were independently hashed: model `b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea`; tokenizer `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.

No model was loaded and no R5 inference request was issued. The binary’s `--version` check in the restricted build shell reports that CUDA devices are not exposed there; this is a build/runtime identity observation, not a quality result.
