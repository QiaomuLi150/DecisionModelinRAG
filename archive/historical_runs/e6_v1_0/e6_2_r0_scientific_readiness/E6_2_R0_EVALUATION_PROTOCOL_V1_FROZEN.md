# E6-2 R0 Evaluation Protocol v1 (Frozen for Owner Review)

## Scope and populations

- Track A: historical QASPER E5 anchor only; no E5-F reopening.
- Track B primary: SciFact EXPLORATORY, 277 qids, source-component split seed 731, after a separately approved authentic Top-20 retrieval freeze. CAL 555 is adapter/development only; RESERVED 277 remains sealed.
- Track C secondary: GaRAGe exact supplied K=15 pools, 2,329 eligible rows; fixed-pool grounding only.

## Inputs and policies

Every model receives identical canonical candidate IDs, ordering and serialized text within a dataset track. No qrels, support labels, answers or outcome-dependent filtering enter model inputs. Eight-model roster is frozen to PPLX 27B, Liquid d1-3B, StartLux-Decision-4B, Amos 2B, Jina v3.5, Qwen3-Reranker-4B, BGE-v2-m3 and Qwen3-Reranker-0.6B. Retrieval Top-5 is a no-reranking control.

## Endpoints

SciFact primary: binary nDCG@5 over judged qrel positives, with Recall@5 and Hit@5 secondary. Report no-positive queries, judged-positive presence in Top-20, and judged-coverage sensitivity. Unjudged documents are not verified negatives even where the conventional evaluator gives zero gain.

GaRAGe secondary: support_any@5 and graded grounding precision@5; UNKNOWN/empty/unjudged values remain explicit missingness.

## Claim restrictions

Dataset differences do not identify K effects. Adapter order diagnostics are not population quality estimates. Quality evidence does not establish serving economics; E6-3 would be required for that. No factual correctness is inferred from these ranking endpoints.
