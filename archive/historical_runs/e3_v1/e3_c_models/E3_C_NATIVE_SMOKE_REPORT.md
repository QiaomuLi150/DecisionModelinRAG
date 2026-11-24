# E3-C Native Smoke Report

{
  "smoke": [
    {
      "model": "liquid_d1_3b",
      "n": 12,
      "valid": 12,
      "context_infeasible": 0,
      "errors": 0,
      "physical_forwards": 12
    },
    {
      "model": "zerank_2",
      "n": 12,
      "valid": 12,
      "context_infeasible": 0,
      "errors": 0,
      "physical_forwards": 36
    }
  ],
  "cal": [
    {
      "model": "liquid_d1_3b",
      "n": 30,
      "valid": 30,
      "context_infeasible": 0,
      "errors": 0,
      "physical_forwards": 30
    },
    {
      "model": "zerank_2",
      "n": 30,
      "valid": 30,
      "context_infeasible": 0,
      "errors": 0,
      "physical_forwards": 90
    }
  ],
  "contract_notes": "All valid rows contain 20 unique canonical IDs, finite scores, deterministic Top-5 membership, gold_in_inference_input=false and native response/input hashes."
}
