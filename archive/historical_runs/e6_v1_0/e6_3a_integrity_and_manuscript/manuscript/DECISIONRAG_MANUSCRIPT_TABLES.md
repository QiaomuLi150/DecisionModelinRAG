# DecisionRAG Model Contract Table

| Model | Family | Revision | Native readout | E6-3A contract status |
|---|---|---|---|---|
| PPLX Decider | general-purpose decision | `14c9d901...` | candidate-aligned choice probabilities | authenticated; full-content corrected run |
| Liquid d1-3B | decision | `051bcc46...` | candidate-aligned choice probabilities | authenticated; full-content corrected run |
| StartLux Decision 4B | decision | `9c0a32e9...` | native decision choice | authenticated; full-content corrected run |
| Amos 1 Small 2B | decision | `09ba3e9d...` | native choice probabilities, max context 16384 | corrected full-content context failures retained |
| Jina v3.5 | specialist | `e8a93f33...` | native listwise relevance scores | authenticated; full-content corrected run |
| Qwen3 Reranker 4B | specialist | `22e68366...` | pairwise/cross-encoder scores | authenticated; full-content corrected run |
| BGE v2-m3 | specialist | `953dc6f6...` | pairwise/cross-encoder scores | authenticated; full-content corrected run |
| Qwen3 Reranker 0.6B | specialist | `e61197ed...` | pairwise/cross-encoder scores | authenticated; full-content corrected run |

See `scientific_synthesis/E6_3A_REPLACEMENT_BOUNDARY_TABLE.csv`, `scientific_synthesis/E6_3A_CROSS_WORKLOAD_METRICS.csv`, and `scientific_synthesis/E6_3A_FIGURE_DATA.csv` for reproducible tables/figure data. Historical R3/R5 scorecards remain the quality source; corrected serving is not a new quality evaluation.
