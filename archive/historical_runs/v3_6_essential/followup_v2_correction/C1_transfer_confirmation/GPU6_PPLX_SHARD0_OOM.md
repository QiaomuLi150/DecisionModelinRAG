# C1 GPU6 PPLX shard-0 resource-gate failure

- Command: `scripts/v3_6_correction_c1_native.py --model pplx --gpu 6 --shard 0 --shards 2 --Ls 6144,8192`
- Outcome: model initialization failed before any C1 request.
- Error: `torch.OutOfMemoryError`; GPU 0 inside the process had only approximately 55 MiB free and an unrelated process held approximately 75.30 GiB.
- Action: no process was terminated and no retry was issued on GPU6.
- Scientific status: no C1 rows were produced by this attempt; the required shard is reassigned to GPU3 after its active Jina run completes.
