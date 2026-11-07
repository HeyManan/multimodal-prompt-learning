"""Base-to-novel targets from MaPLe (Khattak et al., CVPR 2023), Table 3.

These are the published numbers we compare against, not results of this project.
"""

# (dataset, method): (base, new, HM)
PAPER_RESULTS = {
    ("dtd", "ZeroshotCLIP"): (53.24, 59.90, 56.37),
    ("dtd", "CoCoOp"): (77.01, 56.00, 64.85),
    ("dtd", "MaPLe"): (80.36, 59.18, 68.16),
    ("eurosat", "ZeroshotCLIP"): (56.48, 64.05, 60.03),
    ("eurosat", "CoCoOp"): (87.49, 60.04, 71.21),
    ("eurosat", "MaPLe"): (94.07, 73.23, 82.35),
}
