# E2-N N0 Action-Ready Runbook

Current status: `E2_N_N0_READY_AWAITING_GPU_AUTHORIZATION`.

## Verified locally and eligible for a future smoke request

- `Cloudflare/clef-flash`
- `convaiinnovations/laya`
- `BAAI/bge-reranker-v2-m3`
- `Qwen/Qwen3-VL-Reranker-2B`

These are not authorized to run yet. Each still requires a genuine N1 adapter
smoke and exact revision/config/telemetry capture.

## Explicitly blocked or not authorized

- `Bespoke Nimble-9B-v2` — exact snapshot/native dependency unavailable or paid access denied
- `internlm/Intern-Decision-4B` — exact snapshot/native dependency unavailable or paid access denied
- `perplexity-ai/pplx-decider-v1.1-27b` — exact snapshot/native dependency unavailable or paid access denied
- `Qwen/Qwen3-Reranker-0.6B` — exact snapshot/native dependency unavailable or paid access denied
- `Qwen/Qwen3-Reranker-4B` — exact snapshot/native dependency unavailable or paid access denied
- `jinaai/jina-reranker-v3.5` — exact snapshot/native dependency unavailable or paid access denied
- `Alibaba-NLP/gte-reranker-modernbert-base` — exact snapshot/native dependency unavailable or paid access denied
- `cross-encoder/ms-marco-MiniLM-L6-v2` — exact snapshot/native dependency unavailable or paid access denied
- `Cloudflare/clef` — exact snapshot/native dependency unavailable or paid access denied
- `CLM 8B` — exact snapshot/native dependency unavailable or paid access denied
- `Qwen/Qwen3-Reranker-8B` — exact snapshot/native dependency unavailable or paid access denied
- `TypeSafe/Jev` — exact snapshot/native dependency unavailable or paid access denied

## Required owner decision

Authorize E2-N separately by stating: (1) exact GPU IDs and isolation from
existing services, (2) whether public official-checkpoint downloads and network
access are allowed, (3) exact model IDs/revisions within scope, (4) maximum
VRAM/concurrency/wall-time/download limits, and (5) whether any optional or
27B model is included. M1 authorization cannot satisfy this gate.

After authorization, execution order is N1 smoke -> N2 CAL freeze -> N3 full
historical exploratory evaluation -> N4 offline analysis. Failed adapters remain
blocked; no surrogate model or fabricated result may replace them.
