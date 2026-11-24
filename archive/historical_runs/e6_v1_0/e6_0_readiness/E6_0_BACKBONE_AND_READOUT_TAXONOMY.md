# E6-0 Backbone and Readout Taxonomy

| Family | Models | Readout class | Comparability boundary |
|---|---|---|---|
| Decision models | PPLX 27B, Liquid d1-3B, StartLux 4B, Amos Small 2B | typed categorical/noul probabilities or native decision scores | probabilities are not assumed comparable across heads or families; each model selects within its own native view |
| Specialist rerankers | Jina v3.5, Qwen3 4B, BGE v2-m3, Qwen3 0.6B | listwise or pointwise relevance scores/logits | raw score values are not pooled; only rankings over identical candidates are compared |

No architecture-causal claim is permitted. E6-0 freezes only model identity, readout semantics and future adapter requirements; it performs zero inference.
