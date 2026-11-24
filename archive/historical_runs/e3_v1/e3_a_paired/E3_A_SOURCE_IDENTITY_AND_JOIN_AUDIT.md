# E3-A Source Identity and Join Audit

- Canonical population: 4,267 unique query IDs; Track A 3,767 and Track B 500.
- Selection policy-query rows: 4,267 each for retrieval5, BGE, Jina and PPLX; source errors: 0.
- E2-S/P reader rows: 3,300 (300 queries × 11 policies), with 300 unique reader queries; the 120-query/720-case R2 judge subset is nested and separate.
- Reader policy counts: {'retrieval5': 300, 'bge_pointwise': 300, 'gte': 300, 'jina35': 300, 'nimble': 300, 'pplx_decider': 300, 'qwen06': 300, 'qwen4': 300, 'clef_joint': 300, 'minilm': 300, 'qwen_direct': 300}; all statuses `ok`: True; all reader inputs gold-free: True; all application-cache-free: True.
- Reader selection-version mismatches against E2-N2: {}. Jina/PPLX/BGE mismatches are preserved as unmatched rather than reclassified; the historical 231-row Clef mismatch is outside this primary reader comparison.
- Track B qst_0413 remains in the 500-question manifest and is excluded only from identity-dependent scoring under `V11.1R1-GOLD-ID-AMBIGUITY-qst_0413`.
- Gold sidecars were read after frozen selection; no gold field entered any historical model-visible input.
