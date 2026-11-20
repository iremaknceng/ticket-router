Test set: 5528 messages (1004 out-of-scope). Latency: CPU, batch 1, mean of 200 runs on Intel(R) Xeon(R) CPU @ 2.00GHz (2 logical cores).

| Model | Accuracy | In-scope acc. | OOS recall | Appliance acc. | Size (MB) | Latency mean (ms) | p95 (ms) | Threshold |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | 88.75% | 90.10% | 82.67% | 46.43% | 66.8 | 25.35 | 33.46 | 0.25 |
| word2vec + FFNN (own, SGD) | 79.09% | 76.99% | 88.55% | 35.71% | 1.6 | 0.06 | 0.08 | 0.90 |
| mBERT fine-tuned (FP32) | 87.72% | 87.16% | 90.24% | 60.71% | 678.9 | 100.87 | 118.68 | 0.70 |
| Distilled student (FP32) | 89.16% | 89.63% | 87.05% | 67.86% | 516.7 | 52.25 | 64.90 | 0.75 |
| Distilled student (INT8) | 86.05% | 84.64% | 92.43% | 57.14% | 130.3 | 14.05 | 22.26 | 0.80 |
