"""Which images the style distance uses: DomainNet train images and ImageNet validation rows."""
import random
from collections import defaultdict

import pyarrow.parquet as pq

from utils.domainnet import DATASET_FOLDER, read_list
from utils.style_distance import IMAGENET_NUM_CLASSES, IMAGENET_PARQUET_GLOB, IMAGENET_SOURCE

LABEL_COLUMN = "label"
IMAGE_COLUMN = "image"


def sample_style_images(data_root, style, count, seed):
    """Return image paths relative to $DATA, drawn at random from the style's full train list."""
    entries = read_list(data_root / DATASET_FOLDER / "splits" / "{}_train.txt".format(style))
    # Sorting makes the draw independent of the list's order.
    paths = sorted(image_path for image_path, _, _ in entries)
    if len(paths) < count:
        raise ValueError("Expected at least {} {} train images, found {}.".format(count, style, len(paths)))
    # A "<style>-<seed>" string gives every style its own stream, as in prepare_domainnet_splits.py.
    rng = random.Random("{}-{}".format(style, seed))
    return ["{}/{}".format(DATASET_FOLDER, path) for path in rng.sample(paths, count)]


def imagenet_parquet_files(parquet_dir):
    files = sorted(parquet_dir.glob(IMAGENET_PARQUET_GLOB))
    if not files:
        raise FileNotFoundError(
            "Expected ImageNet validation files {} in {}, found none.".format(IMAGENET_PARQUET_GLOB, parquet_dir)
        )
    return files


def index_imagenet_validation(files):
    """Return {label: [(file name, row)]}, reading only the label column."""
    rows_by_label = defaultdict(list)
    for path in files:
        labels = pq.read_table(path, columns=[LABEL_COLUMN]).column(LABEL_COLUMN).to_pylist()
        for row, label in enumerate(labels):
            rows_by_label[label].append((path.name, row))
    if sorted(rows_by_label) != list(range(IMAGENET_NUM_CLASSES)):
        raise ValueError(
            "Expected labels 0-{} in the ImageNet validation files, found {} labels from {} to {}.".format(
                IMAGENET_NUM_CLASSES - 1, len(rows_by_label), min(rows_by_label), max(rows_by_label)
            )
        )
    return rows_by_label


def sample_imagenet_rows(rows_by_label, per_class, seed):
    """Return per_class random (file name, row) pairs for every class, in label order."""
    rng = random.Random("{}-{}".format(IMAGENET_SOURCE, seed))
    sample = []
    for label in range(IMAGENET_NUM_CLASSES):
        rows = rows_by_label[label]
        if len(rows) < per_class:
            raise ValueError(
                "Expected at least {} images of class {}, found {}.".format(per_class, label, len(rows))
            )
        sample += rng.sample(rows, per_class)
    return sample


def imagenet_image_id(file_name, row):
    return "{}:{}".format(file_name, row)


def image_bytes(value, file_name, row):
    if not isinstance(value, dict) or not value.get("bytes"):
        found = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "Expected the Hugging Face image struct with a 'bytes' field in {} row {}, found {}.".format(
                file_name, row, found
            )
        )
    return value["bytes"]


def read_imagenet_images(parquet_dir, samples_by_seed):
    """Return {seed: [JPEG bytes]} in each sample's order.

    Each file is read once for all seeds, and only one file's image column is in memory at a time.
    """
    wanted_rows = defaultdict(set)
    for sample in samples_by_seed.values():
        for file_name, row in sample:
            wanted_rows[file_name].add(row)

    images = {}
    for file_name in sorted(wanted_rows):
        rows = sorted(wanted_rows[file_name])
        column = pq.read_table(parquet_dir / file_name, columns=[IMAGE_COLUMN]).column(IMAGE_COLUMN)
        for row, value in zip(rows, column.take(rows).to_pylist()):
            images[(file_name, row)] = image_bytes(value, file_name, row)
        del column
    return {seed: [images[key] for key in sample] for seed, sample in samples_by_seed.items()}
