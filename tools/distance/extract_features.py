"""Extract frozen CLIP ViT-B/16 image features for the style distance to ImageNet. Needs a GPU.

For each sample seed it draws 5000 random train images per DomainNet style and 5 ImageNet
validation images per class, using the trainers' test transform. Features are L2-normalized in
float64 and cached under $OUTPUT/distance/features/, so a later run reuses them:

    PYTHONPATH="${PWD}" python tools/distance/extract_features.py [--smoke] [--force]
"""
import argparse
import io
import json
import logging

import numpy as np
import torch
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train import extend_cfg  # importing train.py also registers the dataset classes
from utils.clip_model import BACKBONE, load_clip
from utils.distance_measures import normalize_rows
from utils.distance_samples import (
    imagenet_image_id,
    imagenet_parquet_files,
    index_imagenet_validation,
    read_imagenet_images,
    sample_imagenet_rows,
    sample_style_images,
)
from utils.logging_setup import setup_logging
from utils.methods import METHODS
from utils.paths import REPO_ROOT, require_env_dir
from utils.run_record import git_commit
from utils.style_distance import (
    FEATURE_BATCH_SIZE,
    IMAGENET_PARQUET_DIR,
    IMAGENET_SOURCE,
    NUM_WORKERS,
    SAMPLE_SEEDS,
    distance_run,
    feature_path,
    feature_record_path,
)

# The settings that decide the test transform. Train-only augmentations may differ between methods.
TEST_INPUT_KEYS = ("SIZE", "INTERPOLATION", "PIXEL_MEAN", "PIXEL_STD")


class ImageDataset(Dataset):
    """Each source is an image path or JPEG bytes. Images are opened as Dassl's read_image does."""

    def __init__(self, sources, transform):
        self.sources = sources
        self.transform = transform

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        source = self.sources[index]
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        return self.transform(Image.open(source).convert("RGB"))


def method_cfg(config_file):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(str(REPO_ROOT / config_file))
    return cfg


def test_input_settings(cfg):
    settings = {key: cfg.INPUT[key] for key in TEST_INPUT_KEYS}
    settings["normalize"] = "normalize" in cfg.INPUT.TRANSFORMS
    return settings


def build_test_transform():
    """The trainers' test transform, after checking that every method builds the same one."""
    cfgs = {name: method_cfg(method.config_file) for name, method in METHODS.items()}
    first_name = next(iter(cfgs))
    expected = test_input_settings(cfgs[first_name])
    for name, cfg in cfgs.items():
        found = test_input_settings(cfg)
        if found != expected:
            raise RuntimeError(
                "Expected the test input settings of {}, {}, for every method, found {} for {}.".format(
                    first_name, expected, found, name
                )
            )
    return build_transform(cfgs[first_name], is_train=False)


def encode_images(model, sources, transform):
    loader = DataLoader(
        ImageDataset(sources, transform), batch_size=FEATURE_BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    batches = []
    with torch.no_grad():
        for images in loader:
            features = model.encode_image(images.cuda().type(model.dtype))
            batches.append(features.double().cpu().numpy())
    return normalize_rows(np.concatenate(batches))


def needs_features(path, image_ids, force):
    """True if the features must be computed. A cache is reused only if it holds the drawn sample."""
    if force or not path.is_file():
        return True
    cached_ids = np.load(path)["image_ids"].tolist()
    if cached_ids != image_ids:
        raise RuntimeError(
            "Expected the cached features in {} to hold the drawn sample, found other images. "
            "Rerun with --force to recompute them.".format(path)
        )
    logging.info("Reusing %s", path)
    return False


def save_features(run, source, seed, features, image_ids, record):
    path = feature_path(run, source, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record, source=source, sample_seed=seed, count=len(features))
    feature_record_path(run, source, seed).write_text(json.dumps(record, indent=2) + "\n")
    # The .npz marks the features as done, so it must never exist half-written.
    temporary_path = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary_path, features=features, image_ids=np.array(image_ids))
    temporary_path.replace(path)
    logging.info("Wrote %d features to %s", len(features), path)


def extract_imagenet(data_root, run, model, transform, record, force):
    parquet_dir = data_root / IMAGENET_PARQUET_DIR
    rows_by_label = index_imagenet_validation(imagenet_parquet_files(parquet_dir))
    samples = {
        seed: sample_imagenet_rows(rows_by_label, run.imagenet_images_per_class, seed) for seed in run.sample_seeds
    }
    image_ids = {seed: [imagenet_image_id(*row) for row in sample] for seed, sample in samples.items()}
    missing = {
        seed: sample
        for seed, sample in samples.items()
        if needs_features(feature_path(run, IMAGENET_SOURCE, seed), image_ids[seed], force)
    }
    if not missing:
        return
    images = read_imagenet_images(parquet_dir, missing)
    for seed in missing:
        features = encode_images(model, images[seed], transform)
        save_features(run, IMAGENET_SOURCE, seed, features, image_ids[seed], record)


def extract_style(data_root, run, style, seed, model, transform, record, force):
    image_ids = sample_style_images(data_root, style, run.images_per_style, seed)
    if not needs_features(feature_path(run, style, seed), image_ids, force):
        return
    features = encode_images(model, [data_root / image_id for image_id in image_ids], transform)
    save_features(run, style, seed, features, image_ids, record)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--smoke", action="store_true", help="one style, 1000 images, sample seed 1; writes under $OUTPUT/smoke/"
    )
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=list(SAMPLE_SEEDS))
    parser.add_argument("--force", action="store_true", help="recompute features that are already cached")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    data_root = require_env_dir("DATA")
    run = distance_run(require_env_dir("OUTPUT"), args.smoke, args.sample_seeds)
    logging.info("Styles %s, sample seeds %s, features in %s", run.styles, run.sample_seeds, run.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("Expected a usable GPU, found torch.cuda.is_available() == False.")

    transform = build_test_transform()
    logging.info("Test transform: %s", transform)
    model = load_clip().eval().cuda()
    record = {
        "backbone": BACKBONE,
        "transform": str(transform),
        "git_commit": git_commit(),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    logging.info("GPU: %s", record["gpu_name"])

    extract_imagenet(data_root, run, model, transform, record, args.force)
    for style in run.styles:
        for seed in run.sample_seeds:
            extract_style(data_root, run, style, seed, model, transform, record, args.force)
    logging.info("Feature extraction finished.")


if __name__ == "__main__":
    main()
