# E5-0C GPU Authorization and Smoke Reconciliation

E5-0C itself used no GPU, loaded no model, made no HTTP request, and made no inference.

Historical E5-0 synthetic smoke records show PPLX on GPU5 and Jina on GPU6, with one physical forward each. The contemporaneous saved E2-S/P authorization artifact explicitly authorizes GPUs 0/1/2/4 and does not authorize 5/6. The conversation-level instruction that GPUs 5/6 were idle and could be used is not present as an immutable project authorization record. Therefore the historical use is recorded as `GPU_5_6_AUTHORIZATION_UNVERIFIED` at the artifact-governance level; successful execution is not treated as authorization. The raw four smoke rows, including the first two harness false-negative rows, remain preserved.

Future E5-A requires a separate affirmative owner authorization naming exact GPU IDs, current safe/isolated occupancy, device UUIDs, model workload, and no interference with resident services. GPUs 5/6 must not be reused from historical telemetry without that explicit record.
