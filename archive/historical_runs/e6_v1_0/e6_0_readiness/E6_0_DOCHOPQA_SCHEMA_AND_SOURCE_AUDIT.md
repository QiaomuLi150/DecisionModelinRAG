# DocHop-QA Schema and Source Audit

Public metadata was inspected from the official Hugging Face dataset card. The release describes 11,379 QA records, paragraph/table context items, `used_doc`, `context_list`, PMC identifiers, section metadata and raw/content fields. The card reports CC-BY-4.0 and public PMC scientific-document provenance.

## Admission decision

`CONDITIONAL_GRADED_SUPPORT_ONLY`. The archived project does not contain the DocHop-QA release or the original PMC source corpus. A QA row's supplied `context_list` is not silently promoted to the natural retrieval corpus, and the listed context items are not assumed to be an exhaustive necessary-support set. Before E6 comparison, an owner-approved release snapshot, license record, source-corpus reconstruction, paragraph identity join and support semantics audit are required. Until then, complete-support CSSR is not admissible.
