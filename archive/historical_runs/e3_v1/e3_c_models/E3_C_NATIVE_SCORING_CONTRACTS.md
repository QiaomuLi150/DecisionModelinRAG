# E3-C Native Scoring Contracts

Liquid uses the official `system_one(state, questions)` API with one shared text state and twenty named `noul` questions. Each score is the native yes probability. Candidate names are embedded in both the state and the corresponding question; output keys are checked against all twenty canonical positions. This establishes an exploratory ranking interface, not calibrated cross-head global relevance or complementarity.

zerank-2 uses the official Sentence-Transformers `CrossEncoder.predict()` interface over twenty `(question, full_candidate_text)` pairs. It returns raw positive/Yes logits; scores are ranked descending and are never sigmoid-converted or called probabilities. Pair batching and token lengths are recorded per row.
