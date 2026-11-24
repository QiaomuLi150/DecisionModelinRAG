# E5-0 Dataset and License Audit

## QASPER (primary)

The locally archived train/dev v0.3 release contains 1,169 papers, 3,598 questions and 60,148 paragraph units. The authoritative dataset card is https://huggingface.co/datasets/allenai/qasper and reports CC BY 4.0. The downloaded archive URL is https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz. This audit retains the release archive and its README. QASPER is a scientific-document, within-paper QA benchmark. It is not relabeled as cross-document multi-hop. The train/dev archive was downloaded before any E5 selector output and its SHA-256 is in the input inventory. The test archive was not acquired or inspected.

## FRAMES (alternate)

The official benchmark card https://huggingface.co/datasets/google/frames-benchmark was inspected as a candidate multi-article source, but no local source/evidence snapshot was acquired. It is therefore `UNVERIFIED`, not a readiness input.

## MultiHop-RAG (historical diagnostic)

The source repository https://github.com/yixuantt/MultiHop-RAG was already inspected in E3 and is not claimed as an independent E5 source. It remains diagnostic only.

## Legal and data limitations

License metadata is not a legal opinion. QASPER source-paper publisher terms, pretraining exposure, and future redistribution permissions remain separate risks. No claim of training-data exclusion is made.
