# Crystal Property Benchmark Report

## Data Source

- Source: Matbench v0.1 structure datasets hosted by the Materials Project / Matminer.
- Tasks: matbench_mp_gap, matbench_mp_e_form, matbench_mp_is_metal.
- Models: CGCNN, ALIGNN, M3GNet.

## Task Summaries

### matbench_mp_gap

- Task type: regression
- Samples used: 64
- Mean sites per structure: 33.00
| model   |    mae |   rmse |      r2 |
|:--------|-------:|-------:|--------:|
| alignn  | 0.6822 | 0.7795 |  0.6931 |
| m3gnet  | 1.1878 | 1.3023 |  0.1435 |
| cgcnn   | 1.3059 | 1.4886 | -0.1192 |

Best model: alignn

### matbench_mp_e_form

- Task type: regression
- Samples used: 64
- Mean sites per structure: 31.14
| model   |    mae |   rmse |      r2 |
|:--------|-------:|-------:|--------:|
| alignn  | 1.0211 | 1.2797 |  0.0469 |
| m3gnet  | 1.0777 | 1.5463 | -0.3917 |
| cgcnn   | 2.5141 | 2.7868 | -3.5202 |

Best model: alignn

### matbench_mp_is_metal

- Task type: classification
- Samples used: 64
- Mean sites per structure: 30.78
| model   |   accuracy |     f1 |   roc_auc |
|:--------|-----------:|-------:|----------:|
| alignn  |     0.7143 | 0.5    |    1      |
| m3gnet  |     0.8571 | 0.8571 |    0.9167 |
| cgcnn   |     0.7143 | 0.5    |    0.75   |

Best model: alignn
