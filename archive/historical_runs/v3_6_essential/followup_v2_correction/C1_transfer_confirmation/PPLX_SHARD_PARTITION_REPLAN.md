# C1 PPLX L6144/L8192 shard partitioning

The first attempt used two shards and produced incomplete append-only `.part`
files before being stopped for a three-way resource partition. Those partial
files are preserved and are not used for scoring. The same frozen prepared
records, model revision, tokenizer, no-retry policy and label-blind contract
are rerun as three disjoint shards on GPUs 3, 4 and 6. This is a throughput
partition only; no input, model, or analysis rule changed.
