"""Check the committed DomainNet split files and the dataset classes that read them.

Needs the DomainNet lists and images in $DATA, and Dassl. Stops at the first failed check:

    DATA=/path/to/datasets PYTHONPATH="${PWD}" python tools/data/check_domainnet_splits.py
"""
import csv
import json
import logging
from collections import Counter

from dassl.config import get_cfg_default
from dassl.data.datasets import build_dataset
from dassl.utils import read_image

from train import extend_cfg  # importing train.py also registers the dataset classes
from utils.domainnet import (
    DATASET_FOLDER,
    FIRST_NEW_LABEL,
    LAST_BASE_LABEL,
    NUM_CLASSES,
    STYLES,
    TEST_IMAGES_PER_CLASS,
    VAL_IMAGES_PER_CLASS,
    group_of,
    read_all_lists,
)
from utils.logging_setup import setup_logging
from utils.methods import DEFAULT_SEEDS, NUM_SHOTS, SMOKE_TEST_IMAGES_PER_CLASS, ZEROSHOT_RNG_SEED
from utils.paths import (
    DOMAINNET_SPLITS_DIR,
    dataset_config,
    domainnet_classnames_path,
    domainnet_shot_path,
    domainnet_test_path,
    require_env_dir,
)

AUDIT_PATH = DOMAINNET_SPLITS_DIR / "audit.csv"
SMOKE_DATASET = "domainnet_b2n_real_smoke"
NUM_BASE = LAST_BASE_LABEL + 1
NUM_NEW = NUM_CLASSES - NUM_BASE


def require_equal(found, expected, what):
    if found != expected:
        raise RuntimeError("Expected {} to be {}, found {}.".format(what, expected, found))


def require_disjoint(first, second, what):
    shared = set(first) & set(second)
    if shared:
        raise RuntimeError(
            "Expected no images shared by {}, found {}, such as {}.".format(
                what, len(shared), sorted(shared)[0]
            )
        )


def require_subset(paths, allowed, what):
    outside = set(paths) - allowed
    if outside:
        raise RuntimeError(
            "Expected every image of {} in its list, found {} outside it, such as {}.".format(
                what, len(outside), sorted(outside)[0]
            )
        )


def read_json(path):
    return json.loads(path.read_text())


def read_audit():
    with AUDIT_PATH.open(newline="") as file:
        return list(csv.DictReader(file))


def image_paths(items):
    return [item["impath"] for item in items]


def per_class_counts(items):
    return Counter(item["label"] for item in items)


def expected_counts(audit, column, keep):
    """Return {label: images} from an audit column, for the labels with at least one image."""
    counts = {}
    for row in audit:
        count = keep(int(row[column]), int(row["label"]))
        if count > 0:
            counts[int(row["label"])] = count
    return counts


def expected_train(available, label):
    return min(NUM_SHOTS, available) if label <= LAST_BASE_LABEL else 0


def expected_val(available, label):
    if label > LAST_BASE_LABEL:
        return 0
    return min(VAL_IMAGES_PER_CLASS, max(available - NUM_SHOTS, 0))


def expected_test(available, label):
    return min(TEST_IMAGES_PER_CLASS, available)


def check_class_list(class_names, audit):
    require_equal(len(class_names), NUM_CLASSES, "the number of class names")
    require_equal(len(set(class_names)), NUM_CLASSES, "the number of unique class names")
    require_equal([row["classname"] for row in audit], class_names, "the audit's class names")
    for label, row in enumerate(audit):
        require_equal(row["group"], group_of(label), "the group of label {}".format(label))


def check_items(items, class_names, style, what):
    for item in items:
        require_equal(item["classname"], class_names[item["label"]], "a class name in " + what)
        prefix = "{}/{}/".format(DATASET_FOLDER, style)
        if not item["impath"].startswith(prefix):
            raise RuntimeError("Expected {} paths under {}, found {}.".format(what, prefix, item["impath"]))


def listed_paths(entries):
    return {"{}/{}".format(DATASET_FOLDER, image_path) for image_path, _, _ in entries}


def check_style_files(style, audit, class_names, all_entries):
    train_list = listed_paths(all_entries[(style, "train")])
    test_list = listed_paths(all_entries[(style, "test")])

    test = read_json(domainnet_test_path(style))
    check_items(test, class_names, style, "the {} test subset".format(style))
    require_equal(
        per_class_counts(test),
        expected_counts(audit, style + "_test", expected_test),
        "the {} test subset counts".format(style),
    )
    require_equal(len(set(image_paths(test))), len(test), "the unique {} test images".format(style))
    require_subset(image_paths(test), test_list, "the {} test subset".format(style))

    draws = {}
    for seed in DEFAULT_SEEDS:
        what = "{} seed {}".format(style, seed)
        few_shot = read_json(domainnet_shot_path(style, seed))
        train, val = few_shot["train"], few_shot["val"]
        check_items(train + val, class_names, style, what)
        require_equal(
            per_class_counts(train),
            expected_counts(audit, style + "_train", expected_train),
            "the train counts of " + what,
        )
        require_equal(
            per_class_counts(val),
            expected_counts(audit, style + "_train", expected_val),
            "the val counts of " + what,
        )
        require_disjoint(image_paths(train), image_paths(val), "train and val of " + what)
        require_disjoint(image_paths(train + val), image_paths(test), "few-shot and test of " + what)
        require_subset(image_paths(train + val), train_list, what)
        draws[seed] = frozenset(image_paths(train))

    # A style whose classes have at most 16 images could repeat a draw, but no style here does.
    require_equal(len(set(draws.values())), len(DEFAULT_SEEDS), "the distinct draws of " + style)
    logging.info("%s: files match the audit and the official lists", style)
    return draws


def check_styles_differ(draws_by_style):
    for seed in DEFAULT_SEEDS:
        for position, first in enumerate(STYLES):
            for second in STYLES[position + 1:]:
                require_disjoint(
                    draws_by_style[first][seed], draws_by_style[second][seed],
                    "{} and {} at seed {}".format(first, second, seed),
                )


def log_short_classes(audit):
    for style in STYLES:
        column = style + "_train"
        short = [row for row in audit if int(row[column]) < NUM_SHOTS]
        logging.info("%s: %d classes with fewer than %d train images", style, len(short), NUM_SHOTS)
        for row in short:
            logging.info(
                "  %s %s (%s): %s train images",
                row["label"], row["classname"], row["group"], row[column],
            )


def build_style(data_root, dataset, subsample, seed, num_shots):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(str(dataset_config(dataset)))
    cfg.DATASET.ROOT = str(data_root)
    cfg.DATASET.SUBSAMPLE_CLASSES = subsample
    cfg.DATASET.NUM_SHOTS = num_shots
    cfg.SEED = seed
    cfg.freeze()
    return build_dataset(cfg)


def check_built_dataset(dataset, expected_names, expected_test_images, what):
    require_equal(dataset.num_classes, len(expected_names), "num_classes of " + what)
    require_equal(dataset.classnames, expected_names, "the class names of " + what)
    require_equal(dataset.lab2cname, dict(enumerate(expected_names)), "lab2cname of " + what)
    require_equal(len(dataset.test), expected_test_images, "the test images of " + what)
    for datum in dataset.test:
        require_equal(datum.classname, expected_names[datum.label], "a test class name of " + what)


def check_style_classes(data_root, style, class_names):
    dataset_name = "domainnet_b2n_{}".format(style)
    test = read_json(domainnet_test_path(style))
    groups = (
        ("base", class_names[:FIRST_NEW_LABEL], [i for i in test if i["label"] <= LAST_BASE_LABEL]),
        ("new", class_names[FIRST_NEW_LABEL:], [i for i in test if i["label"] > LAST_BASE_LABEL]),
    )
    for subsample, names, test_items in groups:
        for seed, num_shots in ((DEFAULT_SEEDS[0], NUM_SHOTS), (ZEROSHOT_RNG_SEED, -1)):
            what = "{} {} with NUM_SHOTS {}".format(style, subsample, num_shots)
            dataset = build_style(data_root, dataset_name, subsample, seed, num_shots)
            check_built_dataset(dataset, names, len(test_items), what)

    # Dassl converts every image to RGB, which matters for quickdraw's PNG files.
    image = read_image(dataset.test[0].impath)
    require_equal(image.mode, "RGB", "the mode of a {} image".format(style))
    logging.info("%s: dataset class has %d base and %d new classes", style, NUM_BASE, NUM_NEW)


def check_smoke_class(data_root, class_names):
    for subsample, names in (("base", class_names[:FIRST_NEW_LABEL]), ("new", class_names[FIRST_NEW_LABEL:])):
        dataset = build_style(data_root, SMOKE_DATASET, subsample, DEFAULT_SEEDS[0], NUM_SHOTS)
        what = "{} {}".format(SMOKE_DATASET, subsample)
        expected = len(names) * SMOKE_TEST_IMAGES_PER_CLASS
        check_built_dataset(dataset, names, expected, what)
    logging.info("%s: %d test images per class", SMOKE_DATASET, SMOKE_TEST_IMAGES_PER_CLASS)


def main():
    setup_logging()
    data_root = require_env_dir("DATA")
    all_entries = read_all_lists(data_root / DATASET_FOLDER / "splits")
    class_names = read_json(domainnet_classnames_path())
    audit = read_audit()

    check_class_list(class_names, audit)
    draws_by_style = {
        style: check_style_files(style, audit, class_names, all_entries) for style in STYLES
    }
    check_styles_differ(draws_by_style)
    log_short_classes(audit)

    for style in STYLES:
        check_style_classes(data_root, style, class_names)
    check_smoke_class(data_root, class_names)
    logging.info("All DomainNet split checks passed.")


if __name__ == "__main__":
    main()
