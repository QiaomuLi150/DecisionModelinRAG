# E6-4 serving final integrity audit

## Verdict

`E6_3A_CORRECTED_VERSION_AUTHENTICATED_WITH_LIMITATIONS`

The E6-3A audit established that the original E6-3 SciFact serving path passed empty candidate passages because the frozen slate records contained content hashes rather than inline text. The historical rows remain unchanged and are excluded from full-content claims. The corrected E6-3A run reconstructed `title + newline + text` from the authenticated SciFact corpus and retained failures.

The corrected boundary is synchronized native selector scoring with adapter-native serialization/readout. Candidate lookup and persistence are outside the timed region; retrieval and reader generation are not included. Therefore the corrected values are bounded selector diagnostics, not full production request latency or sustainable QPS.

## Corrected request accounting

The authoritative scorecard is `results/non_jev/v2/e6_v1_0/e6_3a_integrity_and_manuscript/serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv`. It reports 3,200 timed requests across eight models and two workloads, 3,144 valid requests and 56 failures. All 56 failures are Amos SciFact context-limit failures (28% of its 200 SciFact attempts); successful-request p95 is not treated as an all-attempt service-level guarantee.

The corrected SciFact medians/p95s include PPLX 923/930 ms and Jina 233/345 ms. The exact per-model values, memory and attempt denominators remain in the CSV. No production QPS or dollar cost is inferred.

## Historical authentication

[
  {
    "path": "results/non_jev/v2/e6_v1_0/e6_3a_integrity_and_manuscript/E6_3A_FINAL_OWNER_REVIEW_GATE.json",
    "exists": true,
    "sha256": "37b18aab67ed4f96c891992694c72ece541b47cecd87707102cee26756eb5716"
  },
  {
    "path": "results/non_jev/v2/e6_v1_0/e6_3a_integrity_and_manuscript/E6_3A_PROVENANCE_MANIFEST.json",
    "exists": true,
    "sha256": "4591609fbcc1d9cf23749681c44e1600e1d8d41e662c047471f94111fcc73779"
  },
  {
    "path": "results/non_jev/v2/e6_v1_0/e6_2_r1_scifact_retrieval_freeze/E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl",
    "exists": true,
    "sha256": "5552c0e605a8d0d23ac3fc7ecd00b7cfe3cfb1eb2e2f1b88a69dd687a292c37d"
  },
  {
    "path": "results/non_jev/v2/e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_PARTITION_MANIFEST.json",
    "exists": true,
    "sha256": "98c81a492c45e1e69b56101b721c5f852e75d10c0cc4bc3fdc8774e2a1c309c8"
  }
]

E6-4 did not modify any historical file.
