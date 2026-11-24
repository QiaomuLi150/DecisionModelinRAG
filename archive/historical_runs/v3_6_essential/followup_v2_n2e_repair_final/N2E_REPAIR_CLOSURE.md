# v3.6 N2-E repaired closure

Status: `PASS_WITH_RESOURCE_CAVEAT`

The primary repaired N2-E record set contains 3240 records: 360 queries × 3 policies × 3 attempts. GPU 3, 4 and 6 came from the clean corrected-run shards; GPU 5 came from the corrected no-fill-in rerun after its first implementation attempt was invalidated.

All raw serving records were label-blind. Frozen BRIGHT labels were joined only in this CPU closure.

The first GPU-5 attempt is preserved separately and excluded because it contained 327 PPLX rows rather than the required 270, caused by synthetic fatal-error fill-in. The corrected GPU-5 shard contains exactly 810 records.

Resource limitation: the current worker process retained a model allocation while the Winnow server was active on some shards. Quality/validity results are retained; those latency and memory observations are descriptive and are not promoted as isolated matched-serving evidence.

N3 remains blocked independently because the exact BM25-only transfer artifact is unavailable; the available v2.1 transfer population is hybrid RRF.
