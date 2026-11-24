# E5-0C Protocol Amendment V1 — Owner Approval

**Date:** 2026-10-14  
**Status:** `APPROVED_PRE_OUTCOME`  
**Scope:** E5-A only; E5-F remains locked.

## Approved amendment

The primary E5-A relevance endpoint is the prespecified set-aware CSSR-U
predicate:

> A query is successful if at least one prespecified, valid, mapped annotated
> support set (cardinality <= 5) is wholly contained in the selected Top-5.

The following rules are unchanged:

- valid support sets are mapped only from the permitted QASPER source records;
- alternative support sets are alternatives, not a mandatory union;
- incomplete or unmappable support annotations remain explicitly flagged and
  are not silently upgraded to complete gold evidence;
- retrieval-infeasible queries remain unconditional failures and are not
  removed from the primary population;
- the frozen exploratory questions, candidate slates, Top-20 ordering and
  source-group assignments are unchanged;
- the original single-wrapper CSSR-U remains a historical descriptive
  diagnostic and sensitivity analysis, not the sole primary endpoint;
- E5-F confirmation remains locked and no reserved outcomes were inspected.

## Approval record

The owner approved this pre-outcome amendment and approved GPU execution for
the next authorized E5-A phase in the current conversation on 2026-10-14.
This record is versioned separately from the historical E5-0 artifacts and
does not alter any previous result.

## Scientific status

`E5_0C_PROTOCOL_AMENDMENT_APPROVED`.

E5-A still requires an execution manifest containing the exact model revisions,
input hashes, candidate-view hashes, selected device UUIDs, telemetry policy,
and physical-call accounting before inference begins.
