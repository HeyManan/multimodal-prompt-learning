**Trend from near to far styles (secondary, NN grouping)**

| Gap | Definition | Far − near | Supported | Spearman ρ (NN distance, mean gap) |
| --- | --- | --- | --- | --- |
| G1 | MaPLe − IVL-12 | -0.23 ± 0.40 | no | -0.43 |
| G1' | MaPLe-12 − IVL-12 | -0.32 ± 0.42 | no | -0.14 |
| G1'' | MaPLe − IVL-9 | -0.57 ± 1.26 | no | -0.43 |
| G2 | MaPLe − CoCoOp | -1.49 ± 0.27 | no | 0.43 |
| G3 | IVL-12 − CoCoOp | -1.26 ± 0.21 | no | 0.43 |
| G4 | MaPLe − MaPLe-12 | 0.08 ± 0.48 | no | -0.37 |
| G5 | IVL-12 − DLP-12 | -0.78 ± 0.54 | no | 0.43 |
| G6 | DLP-12 − CoCoOp | -0.47 ± 0.35 | no | 0.66 |

Near: real, clipart, quickdraw. Far: sketch, painting, infograph.
Groups from the nearest-neighbor ranking in results/distance/near_far.json, with the same 3 near / 3 far rule. Secondary check only: the FD grouping in domainnet_trends.md is primary.
Far − near: per seed, the mean gap over the far styles minus the mean gap over the near styles, in percentage points; mean ± sample SD over 3 seeds.
Supported only if the mean is above 0 and above 2 SD.
Spearman ρ over the 6 styles above between mean NN distance and mean gap is descriptive only; no p-value.
