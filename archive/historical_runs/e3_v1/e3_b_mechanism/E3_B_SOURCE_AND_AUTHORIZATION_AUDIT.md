# E3-B Source and Authorization Audit

Status: `E3_B_PASS_WITH_LIMITATIONS` for the bounded historical experiment; B0/B1/B2/B3 completed under the owner-authorized shared-GPU window. E3-0 remains independent and pending.

- Clef checkpoint: `Cloudflare/clef-flash`, revision `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3`, bfloat16, authenticated E2-M1 native joint-binary contract.
- E2-M1 source rows: 4,267 pointwise and 4,267 joint, all status `ok`, candidate count 20, one physical forward per row, gold-free inputs.
- B0 sample: 400 retrieval-conditional identity-valid historical queries, 100 each HotpotQA/MuSiQue/2Wiki/EnterpriseRAG-Bench; seed 731.
- B1 new inference cap: 2,000 full-slate requests plus 12 smoke and 30 CAL rows. Actual new physical calls: `2102` across `2102` successful and `0` failed rows.
- Host safety result: shared authorized GPU IDs `[0, 1]`; no process was terminated or restarted. The read-only pre-run inventory and per-row GPU attribution are recorded in the provenance manifest/detail trace. GPU 5 was not used.
- E3-A and E3-0 status were preserved; no fresh holdout or Jev/paid API was accessed.

Input SHA256 commitments:
```json
{
  "results/non_jev/v2/e2_m1/E2_M1_JOINT_BINARY_SCORES.jsonl": "5cc6a1bd8c2fe30c308ac40e06505a8762c05bf862052dc402a86289eeb3ecaa",
  "results/non_jev/v2/e2_m1/E2_M1_POINTWISE_SCORES.jsonl": "645e9cf3c42ccfa8b814f790fb86f406085d09039b53347ffa11ddd5d04595d8",
  "results/non_jev/v2/e2_m1/E2_M1_SELECTED_DOCUMENTS.jsonl": "1689e5b126b5c2f4c6127f2e0d349bb022eddf189c3699479f117382349075d8",
  "results/non_jev/v2/e2_n/n3/E2_N_N3_BGE_FULL.jsonl": "645164198c69886f61f1de60e43ccb189ca57146940eb5027326ce9482c4ad5f",
  "results/non_jev/v2/e3_v1/e3_a_paired/E3_A_QUERY_LEVEL_EVIDENCE_STATES.csv": "bc8bbff40ead4284bfd6cede3b9710fe8a49f1c34e1842acc1977b8e426d850c"
}
```
