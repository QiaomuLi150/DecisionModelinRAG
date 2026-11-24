# BRIGHT admission report

- **Dataset:** `xlangai/BRIGHT`, official Hugging Face revision `3066d29c9651a576c8aba4832d249807b181ecae`.
- **Queries:** 1384 across 12 official short-query domains.
- **Admitted document view:** official `documents` configuration, short documents only; `long_documents` is excluded.
- **Candidate protocol:** exact official repository BM25 implementation, then freeze Top-20 before label joins.
- **Known-positive labels:** `gold_ids` are benchmark metadata used only after candidate freeze; they are not sent to models.
- **Reader:** not admitted at this stage; BRIGHT `gold_answer`/reasoning do not by themselves establish an independent factual answer-scoring contract.
- **Historical exposure:** public benchmark exposure cannot be proven absent from pretrained weights; this is recorded as a limitation, not treated as a leak finding.

The authoritative schema/count and repository provenance files are adjacent to this report.
