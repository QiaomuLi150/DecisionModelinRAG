# E4-0 Reference, Cohort and Arm Foundation

Status: `R0_AUTHENTICATED_INPUT_PASS`
Historical E3-D selected MultiHop records audited: 900; eligible for E4-1 selection: 900.
Representative cohort: 200; diagnostic-enriched cohort: 50; seed: 731.
Evidence cells in audited 900-question population: {'NEITHER': 401, 'BOTH_COMPLETE': 319, 'JINA_ONLY': 95, 'PPLX_ONLY': 85}.

The representative cohort is an exposed historical exploratory sample, not a prospective test. The enriched cohort is diagnostic only and is never pooled with representative estimates.

Reference answers and support IDs are held in the offline evaluator/index. They are explicitly excluded from all reader payloads; `gold_in_reader_input` and `answer_in_reader_input` are false for every arm row.

Neutral replacements use a deterministic lexical screen over real frozen candidate passages. This does not prove factual non-contradiction; placebo results remain qualified and no human factual correctness is claimed.

Confirmation status remains `CONFIRMATORY_DATA_NOT_READY / E3_0_QUERY_HELD_OUT_ONLY`; no reserved outcome was opened.

## Row-level audit

{
  "status": "PASS",
  "mapping_failures": 0
}
