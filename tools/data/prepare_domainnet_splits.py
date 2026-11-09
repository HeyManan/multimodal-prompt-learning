"""Write the committed DomainNet split files under splits/domainnet/. Run once; training never does.

Reads only the list files, so it needs no images:

    DATA=/path/to/datasets PYTHONPATH="${PWD}" python tools/data/prepare_domainnet_splits.py

Each generator is seeded with a "<style>-<seed>" string and walks the labels in order. One generator
per class would give classes of equal size the same index pattern, and a plain seed would give
every style the same stream.
"""
import csv
import json
import logging
import random

from utils.domainnet import (
    DATASET_FOLDER,
    FIRST_NEW_LABEL,
    LAST_BASE_LABEL,
    LIST_SPLITS,
    NUM_CLASSES,
    STYLES,
    TEST_IMAGES_PER_CLASS,
    TEST_SUBSET_SEED,
    VAL_IMAGES_PER_CLASS,
    display_name,
    group_of,
    label_to_folder,
    paths_by_label,
    read_all_lists,
)
from utils.logging_setup import setup_logging
from utils.methods import DEFAULT_SEEDS, NUM_SHOTS
from utils.paths import (
    DOMAINNET_SPLITS_DIR,
    domainnet_classnames_path,
    domainnet_shot_path,
    domainnet_test_path,
    require_env_dir,
)

AUDIT_PATH = DOMAINNET_SPLITS_DIR / "audit.csv"
README_PATH = DOMAINNET_SPLITS_DIR / "README.md"
BASE_LABELS = range(LAST_BASE_LABEL + 1)


def build_class_names(all_entries):
    folders = label_to_folder(all_entries)
    names = [display_name(folders[label]) for label in range(NUM_CLASSES)]
    if len(set(names)) != NUM_CLASSES:
        raise ValueError("Expected {} unique names, found {}.".format(NUM_CLASSES, len(set(names))))
    return names


def to_items(paths, label, class_names):
    return [
        {"impath": "{}/{}".format(DATASET_FOLDER, path), "label": label, "classname": class_names[label]}
        for path in paths
    ]


def draw_few_shot(paths, class_names, rng):
    """Draw train and val together, so val always comes from outside the train draw."""
    train, val = [], []
    for label in BASE_LABELS:
        shuffled = list(paths[label])
        rng.shuffle(shuffled)
        val_end = NUM_SHOTS + VAL_IMAGES_PER_CLASS
        train += to_items(shuffled[:NUM_SHOTS], label, class_names)
        val += to_items(shuffled[NUM_SHOTS:val_end], label, class_names)
    return {"train": train, "val": val}


def draw_test_subset(paths, class_names, rng):
    test = []
    for label in range(NUM_CLASSES):
        # painting has no syringe test images, so a label can be absent here.
        available = paths.get(label, [])
        count = min(TEST_IMAGES_PER_CLASS, len(available))
        test += to_items(rng.sample(available, count), label, class_names)
    return test


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def class_counts(all_entries):
    """Return {(style, split): {label: image count}}."""
    return {
        key: {label: len(paths) for label, paths in paths_by_label(entries).items()}
        for key, entries in all_entries.items()
    }


def write_audit(counts, class_names):
    fields = ["label", "classname", "group"] + [
        "{}_{}".format(style, split) for style in STYLES for split in LIST_SPLITS
    ]
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for label, name in enumerate(class_names):
            row = {"label": label, "classname": name, "group": group_of(label)}
            for (style, split), style_counts in counts.items():
                row["{}_{}".format(style, split)] = style_counts.get(label, 0)
            writer.writerow(row)


def log_hyphenated_names(class_names):
    for label, name in enumerate(class_names):
        if "-" in name:
            logging.info("Hyphenated class name: %d %s", label, name)


def log_short_classes(counts, class_names):
    for style in STYLES:
        train_counts = counts[(style, "train")]
        short = [label for label in range(NUM_CLASSES) if train_counts.get(label, 0) < NUM_SHOTS]
        logging.info("%s: %d classes with fewer than %d train images", style, len(short), NUM_SHOTS)
        for label in short:
            logging.info(
                "  %d %s (%s): %d train images, all kept if base",
                label, class_names[label], group_of(label), train_counts.get(label, 0),
            )


def log_short_validation(counts, class_names):
    for style in STYLES:
        train_counts = counts[(style, "train")]
        for label in BASE_LABELS:
            val_count = max(train_counts[label] - NUM_SHOTS, 0)
            if val_count < VAL_IMAGES_PER_CLASS:
                logging.info("%s: %s has %d val images", style, class_names[label], val_count)


def count_groups(items):
    base = sum(1 for item in items if group_of(item["label"]) == "base")
    return base, len(items) - base


def summary_table(all_entries, few_shot_sizes, test_subsets):
    lines = [
        "| Style | Train list | Test list | Shot train | Val | Test subset base | Test subset new |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for style in STYLES:
        train, val = few_shot_sizes[style]
        base, new = count_groups(test_subsets[style])
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                style, len(all_entries[(style, "train")]), len(all_entries[(style, "test")]),
                train, val, base, new,
            )
        )
    return "\n".join(lines)


def gap_lines(counts, class_names):
    lines = []
    for (style, split), style_counts in counts.items():
        for label in range(NUM_CLASSES):
            if style_counts.get(label, 0) == 0:
                lines.append(
                    "- {} {}: no images for {} (label {}, {} class)".format(
                        style, split, class_names[label], label, group_of(label)
                    )
                )
    return "\n".join(lines)


def write_readme(all_entries, counts, class_names, few_shot_sizes, test_subsets):
    text = """# DomainNet split files

Generated by `tools/data/prepare_domainnet_splits.py`. Do not edit by hand.

All styles share one class split: labels 0-{last_base} are base and {first_new}-{last_label} are
new, in the label order of the official cleaned lists. `classnames.json` holds the {num_classes}
names in that order. Few-shot files hold base classes only. Train and val sizes are the same for
seeds {seeds}. Val comes from train, outside the {shots}-shot draw, and never from test.

`test_subset.json` holds up to {test_images} images per class, drawn once, and is the same for every
method and seed. Per-class counts are in `audit.csv`.

{table}

## Classes with no images in a list

{gaps}

Both are new classes, so neither affects base training. Each class stays in the fixed class list.
""".format(
        last_base=LAST_BASE_LABEL,
        first_new=FIRST_NEW_LABEL,
        last_label=NUM_CLASSES - 1,
        num_classes=NUM_CLASSES,
        seeds=", ".join(str(seed) for seed in DEFAULT_SEEDS),
        shots=NUM_SHOTS,
        test_images=TEST_IMAGES_PER_CLASS,
        table=summary_table(all_entries, few_shot_sizes, test_subsets),
        gaps=gap_lines(counts, class_names),
    )
    README_PATH.write_text(text)


def main():
    setup_logging()
    data_root = require_env_dir("DATA")
    all_entries = read_all_lists(data_root / DATASET_FOLDER / "splits")
    class_names = build_class_names(all_entries)
    counts = class_counts(all_entries)

    log_hyphenated_names(class_names)
    log_short_classes(counts, class_names)
    log_short_validation(counts, class_names)

    write_json(domainnet_classnames_path(), class_names)
    write_audit(counts, class_names)

    few_shot_sizes, test_subsets = {}, {}
    for style in STYLES:
        train_paths = paths_by_label(all_entries[(style, "train")])
        for seed in DEFAULT_SEEDS:
            rng = random.Random("{}-{}".format(style, seed))
            few_shot = draw_few_shot(train_paths, class_names, rng)
            write_json(domainnet_shot_path(style, seed), few_shot)
            few_shot_sizes[style] = (len(few_shot["train"]), len(few_shot["val"]))

        test_paths = paths_by_label(all_entries[(style, "test")])
        rng = random.Random("{}-{}".format(style, TEST_SUBSET_SEED))
        test_subsets[style] = draw_test_subset(test_paths, class_names, rng)
        write_json(domainnet_test_path(style), test_subsets[style])
        logging.info(
            "%s: %d train, %d val per seed, %d test images",
            style, few_shot_sizes[style][0], few_shot_sizes[style][1], len(test_subsets[style]),
        )

    write_readme(all_entries, counts, class_names, few_shot_sizes, test_subsets)
    logging.info("Wrote the split files to %s", DOMAINNET_SPLITS_DIR)


if __name__ == "__main__":
    main()
