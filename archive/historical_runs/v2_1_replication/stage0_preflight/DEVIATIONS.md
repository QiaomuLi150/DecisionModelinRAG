# DEVIATIONS

- Stage 0 stopped under A3-R2 because neither workload has the exact required existing BM25-only Top-20 artifact.
- The historical v1.1 hybrid RRF Top-20 traces were inspected only for availability and identity diagnostics; they were not substituted for the frozen input population.
- No new retrieval, selector inference, timing, statistical ranking evaluation, GPU use, API request, retry or fill-in occurred.
- The MuSiQue tokenizer length scan is marked diagnostic-only because its source slate is hybrid RRF rather than the required BM25 slate. Enterprise length scanning is not admissible because the exact candidate view adapter is absent.
