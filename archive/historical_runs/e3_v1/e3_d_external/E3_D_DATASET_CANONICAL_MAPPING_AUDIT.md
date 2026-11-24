# E3-D Dataset Canonical Mapping Audit

Status: `CANDIDATE_CONTRACT_PASS`.

- Frozen exploratory questions: **1,009** ({'financebench': 109, 'multihop': 900}), selected before model outcomes.
- Candidate rows: **1,009**.
- Every row has exactly 20 unique candidates: **True**.
- Retrieval pool: BM25 top-100 plus Qwen3-VL-Embedding-2B dense top-100, RRF k=60.
- Candidate generation accessed gold labels: **false**.
- Reserved IDs accessed: **false**.

MultiHop-RAG candidates are article-level documents identified by normalized source URL hashes. FinanceBench candidates are real PDF pages identified by `doc_name:p{zero_based_page}`; page extraction is recorded in `E3_D_CORPUS_PREPARATION.json`.

FinanceBench page annotations are used only after candidate generation for diagnostic/reference scoring; they are not assumed exhaustive.
