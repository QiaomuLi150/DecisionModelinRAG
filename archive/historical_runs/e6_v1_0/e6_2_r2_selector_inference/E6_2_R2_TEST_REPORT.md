# E6-2 R2 Test Report

- Focused R2 integrity tests: **11/11 passed**.
- Full authoritative regression suite: **225/225 passed** in the shared MultiModalRAG environment.
- Native selector attempts: 2,216 scheduled; 1,939 executed because StartLux was explicitly blocked at model load by missing `causal_conv1d`.
- Amos: 277 attempts, 194 valid, 83 explicit context-limit failures.
- Quality scoring, reader, Jev and paid API calls: **0**.
