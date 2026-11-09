"""Load the frozen CLIP model the way the trainers do."""
from dassl.config import get_cfg_default

from trainers.coop import load_clip_to_cpu

BACKBONE = "ViT-B/16"


def load_clip():
    # clip.load() fails in this repository: it does not pass the design_details
    # argument that build_model() requires. The trainers load CLIP this way instead.
    cfg = get_cfg_default()
    cfg.MODEL.BACKBONE.NAME = BACKBONE
    return load_clip_to_cpu(cfg)
