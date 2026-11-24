#!/usr/bin/env python3
"""Repair V0's historical ledger paths without rereading the CRAG corpus."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e11v_v1_0/00_inventory"
HIST = [
    ROOT / "results/non_jev/v2/e2_m0/E2_M0_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e2_m1/E2_M1_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol/E5_0_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e5_v1_1/e5_0c_readiness_closure/E5_0C_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e5_v1_1/e5_a_replacement_benchmark/E5_A_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e6_v1_0/e6_0_readiness/E6_0_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e6_v1_0/e6_1_native_adapter_validation/E6_1_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e7_v1_0/04_closure/E7_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e7_v1_0/04_closure/E7_FINAL_OWNER_REVIEW_GATE.json",
    ROOT / "results/non_jev/v2/e8_v1_0/04_closure/E8_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e8_v1_0/04_closure/E8_FINAL_OWNER_REVIEW_GATE.json",
    ROOT / "results/non_jev/v2/e9_bright_v1_0/06_closure/E9_BRIGHT_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e9_bright_v1_0/06_closure/E9_BRIGHT_FINAL_OWNER_REVIEW_GATE.json",
    ROOT / "results/non_jev/v2/e10_v1_0/05_closure/E10_FINAL_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e10_v1_0/05_closure/E10_FINAL_OWNER_REVIEW_GATE.json",
    ROOT / "results/non_jev/v2/e11_v1_0/07_closure/E11_FINAL_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e11_v1_0/07_closure/E11_FINAL_OWNER_REVIEW_GATE.json",
    ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_BUDGET_POLICY_CONFIG.json",
]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
records=[]
for p in HIST:
    x={'path':str(p),'exists':p.exists()}
    if p.exists(): x.update(bytes=p.stat().st_size,sha256=digest(p))
    records.append(x)
doc=json.loads((OUT/'V0_SOURCE_MANIFEST.json').read_text())
doc['source_manifest']=records
doc['historical_artifacts_hashed']=sum(x['exists'] for x in records)
doc['historical_artifact_missing']=sum(not x['exists'] for x in records)
(OUT/'V0_SOURCE_MANIFEST.json').write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')
(OUT/'V0_HISTORICAL_LEDGER_AUDIT.md').write_text('# E11-V V0 Historical Ledger Audit\n\nCPU/offline audit only. No model, reader, Jev, paid API, GPU, or protected-holdout calls were made. Existing files were not modified.\n\n'
  +'\n'.join(f"- `{x['path']}` — exists={x['exists']}, bytes={x.get('bytes')}, sha256={x.get('sha256')}" for x in records)
  +'\n\nE11 frozen policy: `E11-A-EQUAL-PREFIX-128-MARGIN-V1`; PPLX revision `14c9d90137962d300f3de95633f7b6535e80c5ce`; Jina revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc`.\n')
print(json.dumps({'hashed':sum(x['exists'] for x in records),'missing':sum(not x['exists'] for x in records)}))
