# E6-2 R4C GaRAGe dataset and eligibility audit

## Authenticated release

- Release path: `data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl`
- Rows: 2366; unique sample IDs: 2366
- Exact K=15, annotation-aligned rows: 2329
- Excluded from structural eligibility: 37 (non-K15 or misaligned)
- Dataset SHA-256: `419e3941f6e8eb4082a74ca2140c1f9337f8b467ff76656a6b8b0290ca3f3a72`
- LICENSE SHA-256: `41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0`; README SHA-256: `b3f7ab1186a759746e5f80cb0823aea07743acef91dedc416292cd154965a397`
- Candidate identity: `sample_id:cite_i` positional; original order preserved.
- Source-document IDs/URLs: absent in the local release schema.

## Annotation semantics

`evidence_relevant` is binary YES/NO and is the only primary relevance label. `evidence_correct` is categorical and remains separate; `evidence_cited` is citation-related. Their observed counts over the eligible K15 rows are `{'YES': 25663, 'NO': 9268, '': 4}`. No categorical values were collapsed into ordinal grades and no alternative evidence labels were unioned into a mandatory set.

The eligible population is structural only. No model outcome, relevance label, answer, or citation field was used for eligibility or partition allocation.
