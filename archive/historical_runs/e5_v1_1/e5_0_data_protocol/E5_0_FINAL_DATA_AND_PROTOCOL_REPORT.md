# DecisionRAG E5-0 Final Data and Protocol Report

## Gate

**E5_0_CONDITIONAL_READY**. QASPER train/dev is an admissible text-only primary exploratory source with paragraph-level designated evidence and source-document grouping. The result is conditional because external reservation attestation and independent reserved-access isolation are unavailable. E5-F remains locked.

## Data

QASPER: 1169 papers, 3598 questions, 60,148 paragraphs in the downloaded train/dev snapshot. Eligible mapped 1–5 paragraph supports on permitted partitions: 2021; CAL=519, E5_EXPLORATORY=1502. Mapping exclusions are in the CSV and annotation report. QASPER is within-paper, not cross-document multi-hop. FRAMES is unverified and MultiHop-RAG is historical diagnostic only.

## Source independence and reservation

1169 exact-content source components were assigned by frozen source-group hashing. No question-level split was used. Across all source groups, question-size max/median/p90 are 12/3/5, with largest-group fraction 0.0033. The reserved partition has 292 groups and 867 questions, represented only by hashes/counts. Local chronology is recorded; external attestation is unavailable and access isolation is partial.

## Retrieval

Natural Top-20 slates were built only for eligible E5_EXPLORATORY questions with BM25 + Qwen embedding + RRF(60), no gold/qrels in requests. Candidate freeze hash is `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`. The dense stage used a local exact cosine matrix and did not modify Qdrant; this is a candidate-freeze implementation detail, not production ANN timing evidence.

## Future contract

Primary E5-A is frozen as equal-source-group CSSR-U PPLX−Jina with 5,000 group bootstrap replicates, seed 731, two-sided 95% intervals, CSSR-C and query-weighted sensitivity. Contracts, invalid-output handling, reader validation and serving protocol are frozen but not executed.

## Not performed

No benchmark selector/reader inference, E5-A/B/C/D/F, Jev, paid API, training, new model family, multimodal experiment, or holdout outcome access was performed. Only the SDD-authorized synthetic, label-free Jina/PPLX interface smoke was run; it produced no benchmark quality result.
