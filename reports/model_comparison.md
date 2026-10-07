# Test-period results (2014-05-07 to 2020-04-01)

Recurrent rows are mean ± std over 3 seeds. Always predicting 'up' would score 52.8% directional accuracy.

| Model | Skill vs naive (%) | Directional Acc (%) | Return RMSE (%) | Price RMSE | Price MAPE (%) | Price R2 |
|---|---|---|---|---|---|---|
| Naive (tomorrow = today) | 0.00 | — | 1.729 | 3.29 | 1.16 | 0.9962 |
| Drift (average return) | 0.16 | 52.76 | 1.728 | 3.29 | 1.16 | 0.9962 |
| Linear AR (ridge) | -0.70 | 51.48 | 1.736 | 3.30 | 1.17 | 0.9962 |
| LSTM | -0.03 ± 0.08 | 50.31 ± 0.39 | 1.730 ± 0.001 | 3.27 ± 0.00 | 1.17 ± 0.00 | 0.9962 ± 0.0000 |
| GRU | -0.16 ± 0.48 | 50.29 ± 1.02 | 1.731 ± 0.004 | 3.29 ± 0.02 | 1.17 ± 0.00 | 0.9962 ± 0.0000 |
| Original: GRU on raw prices | -347,341 | 47.24 | 101.944 | 112.12 | 60.25 | -3.4329 |
