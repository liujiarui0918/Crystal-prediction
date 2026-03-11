# Crystal Property Benchmark Report

## Data Source

- Source: Matbench v0.1 structure datasets hosted by the Materials Project / Matminer.
- Tasks: matbench_mp_gap, matbench_mp_e_form, matbench_mp_is_metal.
- Models: CGCNN, ALIGNN, M3GNet.

## Task Summaries

### matbench_mp_gap

- Task type: regression
- Samples used: 8
- Mean sites per structure: 38.75
| model   |    mae |   rmse | r2   |
|:--------|-------:|-------:|:-----|
| cgcnn   | 0.1054 | 0.1054 |      |
| alignn  | 1.0575 | 1.0575 |      |
| m3gnet  | 1.8308 | 1.8308 |      |

Best model: cgcnn

### matbench_mp_e_form

- Task type: regression
- Samples used: 8
- Mean sites per structure: 46.00
| model   |    mae |   rmse | r2   |
|:--------|-------:|-------:|:-----|
| m3gnet  | 0.4164 | 0.4164 |      |
| alignn  | 0.9579 | 0.9579 |      |
| cgcnn   | 1.8443 | 1.8443 |      |

Best model: m3gnet

### matbench_mp_is_metal

- Task type: classification
- Samples used: 8
- Mean sites per structure: 24.50
| model   |   accuracy |   f1 | roc_auc   |
|:--------|-----------:|-----:|:----------|
| cgcnn   |          0 |    0 |           |
| alignn  |          0 |    0 |           |
| m3gnet  |          0 |    0 |           |

Best model: cgcnn
