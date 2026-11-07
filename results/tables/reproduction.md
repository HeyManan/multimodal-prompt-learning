| Dataset | Method | Seeds | Base | Base paper | Base diff | New | New paper | New diff | HM | HM paper | HM diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dtd | ZeroshotCLIP | n/a | 53.24 | 53.24 | +0.00 | 60.51 | 59.90 | +0.61 | 56.64 | 56.37 | +0.27 |
| dtd | CoCoOp | 3 | 77.24 ± 0.07 | 77.01 | +0.23 | 53.86 ± 2.84 | 56.00 | -2.14 | 63.47 ± 1.95 | 64.85 | -1.38 |
| dtd | MaPLe | 3 | 80.98 ± 0.64 | 80.36 | +0.62 | 57.45 ± 7.12 | 59.18 | -1.73 | 67.21 ± 5.07 | 68.16 | -0.95 |
| eurosat | ZeroshotCLIP | n/a | 56.79 | 56.48 | +0.31 | 64.00 | 64.05 | -0.05 | 60.18 | 60.03 | +0.15 |
| eurosat | CoCoOp | 3 | 87.82 ± 1.20 | 87.49 | +0.33 | 63.90 ± 14.57 | 60.04 | +3.86 | 73.97 ± 10.61 | 71.21 | +2.76 |
| eurosat | MaPLe | 3 | 89.81 ± 5.41 | 94.07 | -4.26 | 69.94 ± 7.50 | 73.23 | -3.29 | 78.64 ± 4.86 | 82.35 | -3.71 |

Accuracy in %, mean ± sample SD over seeds. Diff is ours minus the paper's Table 3.
Mean HM is computed from the averaged base and new accuracies.
(prelim.) marks fewer than 3 seeds. Seeds n/a marks a run with no training, which has no seeds.
