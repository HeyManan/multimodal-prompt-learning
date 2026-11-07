"""Fixed DomainNet facts shared by the split tools and the dataset classes."""
from collections import defaultdict

STYLES = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
LIST_SPLITS = ("train", "test")
NUM_CLASSES = 345
LAST_BASE_LABEL = 172
FIRST_NEW_LABEL = LAST_BASE_LABEL + 1
VAL_IMAGES_PER_CLASS = 4
TEST_IMAGES_PER_CLASS = 50
TEST_SUBSET_SEED = 0


def group_of(label):
    return "base" if label <= LAST_BASE_LABEL else "new"


def display_name(folder_name):
    """Class name given to the model. Hyphens stay, so 't-shirt' keeps its usual spelling."""
    return folder_name.replace("_", " ").lower()


def read_list(list_path):
    """Return (image path, class folder, label) for every line of a DomainNet list file."""
    if not list_path.is_file():
        raise FileNotFoundError("Expected a DomainNet list at {}, found no file.".format(list_path))
    entries = []
    for line in list_path.read_text().splitlines():
        image_path, label = line.split()
        entries.append((image_path, image_path.split("/")[1], int(label)))
    return entries


def read_all_lists(split_dir):
    return {
        (style, split): read_list(split_dir / "{}_{}.txt".format(style, split))
        for style in STYLES
        for split in LIST_SPLITS
    }


def label_to_folder(all_entries):
    """Map each label to its class folder, using every list so no style's gaps can drop a class."""
    names = {}
    for entries in all_entries.values():
        for _, folder, label in entries:
            if names.setdefault(label, folder) != folder:
                raise ValueError(
                    "Expected one class folder for label {}, found {} and {}.".format(
                        label, names[label], folder
                    )
                )
    if sorted(names) != list(range(NUM_CLASSES)):
        raise ValueError("Expected labels 0-{}, found {} labels.".format(NUM_CLASSES - 1, len(names)))
    return names


def paths_by_label(entries):
    """Return {label: sorted image paths}. Sorting makes a draw independent of the list's order."""
    grouped = defaultdict(list)
    for image_path, _, label in entries:
        grouped[label].append(image_path)
    return {label: sorted(paths) for label, paths in grouped.items()}
