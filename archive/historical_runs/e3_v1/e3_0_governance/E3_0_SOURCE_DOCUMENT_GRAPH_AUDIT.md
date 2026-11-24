# E3-0 Source-Document Graph Audit

Seed: `731`.

Partition construction used source-document metadata only. Answer strings, evidence text, model scores and reader outputs were not used.

## MultiHop-RAG

- Query records: **2,556**.
- Valid identity records: **2,255**.
- Source IDs: **609**.
- Connected components: **392**.
- Largest component: **1,895 queries**.
- Component-size distribution: `{1: 322, 2: 29, 3: 14, 4: 5, 5: 5, 6: 2, 7: 4, 8: 2, 9: 1, 10: 2, 11: 1, 15: 1, 16: 1, 24: 1, 43: 1, 1895: 1}`.
- Identity mapping failures: **0**.
- Metadata rule: evidence_list source/title/url metadata only; answer strings not used for partition

## FinanceBench

- Query records: **150**.
- Valid identity records: **150**.
- Source IDs: **84**.
- Connected components: **84**.
- Largest component: **7 queries**.
- Component-size distribution: `{1: 52, 2: 14, 3: 12, 4: 2, 5: 1, 7: 3}`.
- Identity mapping failures: **0**.
- Metadata rule: doc_name/evidence document/company/period metadata only; answer and evidence text not used for partition

## Reservation result

Reserved query IDs: **195**; exploratory query IDs: **2,511**.
Reserved/exploratory overlap: **0**.
Every query in a connected source-document component received one assignment; no component was split across partitions.

The reservation is analysis-held-out and does not prove that public model pretraining did not contain these records.
