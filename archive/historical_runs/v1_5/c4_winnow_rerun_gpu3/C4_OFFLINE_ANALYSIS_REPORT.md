# v1.5 C4 GPU3 Offline Analysis

- Raw rows: `1384`; valid native selections: `1045`; failures retained: `339`.
- Operational all-attempt Winnow known-positive nDCG@5: `0.142636`.
- Jina full reference all-attempt nDCG@5: `0.184369`.
- Winnow minus Jina query-weighted estimate: `-0.041733`, 95% CI `-0.05549507001375727` to `-0.02874191936440529`.
- Equal-domain sensitivity is in `export/c4_summary.json`.

The raw trace was label-blind; labels were joined only by this offline command after raw-output completion. Native failures score zero in the operational all-attempt metric and remain separately reported. This is a v1.5 control result, not a production replacement claim.
