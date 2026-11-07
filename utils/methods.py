"""Registry of the compared methods. A new method only needs a new METHODS entry."""
from typing import NamedTuple, Optional

NUM_SHOTS = 16
DEFAULT_SEEDS = (1, 2, 3)
SMOKE_EPOCHS = 1

# Recorded in results.json and used in folder names for runs without seeds.
NO_SEED = "none"
# Zero-shot evaluation has no random step, so this value cannot change its accuracy.
# It is set only so that every run has a seed that is set and logged.
ZEROSHOT_RNG_SEED = 1


class Method(NamedTuple):
    trainer: str
    config_file: str
    final_epoch: Optional[int]  # None means the method has no training and no seeds

    @property
    def is_trained(self):
        return self.final_epoch is not None


METHODS = {
    "MaPLe": Method(
        trainer="MaPLe",
        config_file="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml",
        final_epoch=5,
    ),
    "CoCoOp": Method(
        trainer="CoCoOp",
        config_file="configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml",
        final_epoch=10,
    ),
    "ZeroshotCLIP": Method(
        trainer="ZeroshotCLIP",
        config_file="configs/trainers/CoOp/vit_b16.yaml",
        final_epoch=None,
    ),
}


def get_method(name):
    if name not in METHODS:
        raise ValueError("Expected a method in {}, found '{}'.".format(sorted(METHODS), name))
    return METHODS[name]
