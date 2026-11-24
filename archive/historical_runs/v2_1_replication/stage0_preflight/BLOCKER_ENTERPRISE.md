# EnterpriseRAG-Bench blocker

Status: `BLOCKED` under decision rule A3-R1.

The repository contains 500 questions, 443 `APPLICABLE` identity-resolved quality rows, the official archive and a prior 500-row V11 retrieval trace. That trace stores a hybrid `BM25 top-100 + dense top-100 + RRF@60` Top-20, not an existing exact BM25-only Top-20 slate. The prior `candidate_order` is not a BM25-only order. Although BM25 top-100 lists are embedded in the trace, deriving and using their first 20 would create a new candidate population and is prohibited by the frozen replication contract. The prior chunk manifest also does not itself provide the exact selector input strings.

No model call was made. Unblocking requires an existing hashable BM25 Top-20 slate plus its exact candidate text adapter, or an owner-approved versioned protocol amendment.
