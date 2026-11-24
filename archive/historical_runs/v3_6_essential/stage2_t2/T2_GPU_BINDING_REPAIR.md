# T2 GPU-binding repair

The first T2-A launch was stopped before any request-level inference. The
wrapper imported the native formatter before setting `CUDA_VISIBLE_DEVICES`,
so the 27B loader could begin on the wrong physical device. GPU3 remained at
its preflight level and the process was interrupted during model transfer.

The process exited with code 130. No T2 raw request trace was written and no
label or protected holdout was accessed. The host GPU snapshot after
termination showed the changed allocation had returned to its pre-existing
level. The repair sets `CUDA_VISIBLE_DEVICES=<requested physical GPU>` before
the processor and native formatter imports, then uses logical `cuda:0` inside
the isolated process. The corrected run is versioned in the T2 session
namespace; this incident is retained as a non-authoritative pre-inference
attempt.
