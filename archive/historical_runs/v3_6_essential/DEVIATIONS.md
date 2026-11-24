# v3.6 deviations

- Three T1 session namespaces were scheduled on GPU3, GPU6 and a second GPU3 namespace. Two namespaces contain the complete five-policy primary set; the third contains only the three Jina policies because GPU3 was released after that block. Therefore the T1 serving gate remains partial at 2/3 complete replicate sessions. The third session is sequential rather than concurrent with the first GPU3 session; same-session ratios remain valid, but cross-session interference is not a concurrent three-GPU estimate.
- The initial Jina two-stage wrapper emitted a tuple-key error. Its raw failures are preserved; the affected policy was rerun in the versioned `jina_j2_repair` namespace and only the repaired trace is authoritative for timing.
- Historical Winnow, Drex and Liquid records were not newly timed by this runner; their quality records remain historical controls.
- Solo C120, T2 frontier sessions, determinism rechecks, and optional Stage 4 remain pending/not run.
- No labels were read by the serving runner; quality labels were joined only in the CPU analysis stage.
