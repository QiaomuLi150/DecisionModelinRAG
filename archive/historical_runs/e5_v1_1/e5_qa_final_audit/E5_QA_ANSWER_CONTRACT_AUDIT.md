# E5-QA Answer Contract Audit

Verdict: `REFERENCE_PROXY_AUDITED` with important validity limits.

The QASPER snapshot contains answer annotations with free-form/abstractive, extractive, yes/no and unanswerable fields. Per-question family counts over the 1,502 E5-A records are `{'ABSTRACTIVE': 312, 'AMBIGUOUS_REFERENCE': 30, 'EXTRACTIVE': 862, 'MIXED_ANNOTATED': 104, 'YES_NO': 194}`; reference statuses are `{'AMBIGUOUS_REFERENCE': 30, 'ANSWERABLE': 1472}`. Annotation-count distribution is `{1: 1126, 2: 372, 3: 4}`. Alternative answer annotations and evidence strings were preserved as alternatives; they were not unioned into one mandatory answer or support set.

The inherited reader contract is text-only and asks for one short answer, explicitly permits `INSUFFICIENT_EVIDENCE`, supplies five selected passages, caps each passage view at 256 tokenizer tokens, and uses deterministic decoding (temperature 0, top_p 1, max_tokens 64, seed 0). It does not require citations. The raw reader rows preserve prompt hashes, selected IDs, passage-view hashes, request hashes, response hashes and token counts. 78/496 physical reader rows included at least one truncated passage view. Representative complete-support Top-20 feasibility is 45/200; this is a ceiling/availability diagnostic, not a correctness label.

The low frozen match rate is therefore compatible with several unresolved causes: free-form paraphrase and scientific terminology, answer-type mismatch, numeric/unit formatting, forced short output, abstention behavior, five-passage truncation and absent supporting passages. It cannot be converted into a factual-error rate.

Offline symmetric rescoring variants and the diagnostic casebook are separate post-hoc analyses. No reader or model was rerun.
