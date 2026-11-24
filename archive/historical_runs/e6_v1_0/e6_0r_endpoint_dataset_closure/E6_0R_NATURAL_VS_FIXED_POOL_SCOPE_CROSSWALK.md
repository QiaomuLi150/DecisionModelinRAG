# Natural vs Fixed-Pool Scope Crosswalk

| Track | Corpus construction | Candidate K | Primary interpretation |
|---|---|---:|---|
| N-REL SciFact | natural 5,183-row corpus; future gold-blind retrieval | Top-20→Top-5 | document-level qrel relevance; sparse judging |
| F-GRD GaRAGe | question-associated supplied pool | K=15→5 | grounding/support labels only; fixed-pool |
| Historical QASPER | prior frozen experiment | Top-20→Top-5 | complete-support CSSR-U; unchanged |

No metric, denominator or source-group CI is pooled across these tracks.
