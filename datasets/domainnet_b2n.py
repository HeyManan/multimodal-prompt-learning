import json
import os
from collections import Counter

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from utils.domainnet import FIRST_NEW_LABEL, LAST_BASE_LABEL
from utils.methods import NUM_SHOTS, SMOKE_TEST_IMAGES_PER_CLASS
from utils.paths import domainnet_classnames_path, domainnet_shot_path, domainnet_test_path


def read_split_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Expected the DomainNet split file {}, found none. It is committed to the "
            "repository and comes from tools/data/prepare_domainnet_splits.py.".format(path)
        )
    return json.loads(path.read_text())


class DomainNetB2NBase(DatasetBase):
    """One DomainNet style in the base-to-novel setting.

    Not registered: each style below is a registered class. Dassl's own DomainNet class has
    another layout and uses test as val, and a class of that name is already registered.

    OxfordPets.subsample_classes cannot be used, because it infers the split from train labels.
    painting has no t-shirt in train, so it would move the boundary for that style.
    """

    style = None

    def __init__(self, cfg):
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        if subsample not in ("base", "new"):
            raise ValueError(
                "Expected DATASET.SUBSAMPLE_CLASSES to be base or new, because the few-shot "
                "files hold only base classes, found '{}'.".format(subsample)
            )
        if cfg.DATASET.NUM_SHOTS not in (NUM_SHOTS, -1):
            raise ValueError(
                "Expected DATASET.NUM_SHOTS to be {} or unset, found {}.".format(
                    NUM_SHOTS, cfg.DATASET.NUM_SHOTS
                )
            )

        self.data_root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.all_classnames = read_split_file(domainnet_classnames_path())

        if subsample == "base":
            first_label, last_label = 0, LAST_BASE_LABEL
        else:
            first_label, last_label = FIRST_NEW_LABEL, len(self.all_classnames) - 1

        # Zero-shot also reads this file, for the placeholder below. Its fixed RNG seed
        # is a seed that has a file.
        few_shot = read_split_file(domainnet_shot_path(self.style, cfg.SEED))
        # Dassl builds a train loader even with --eval-only and asserts it is not empty, but
        # never reads it. So new runs and zero-shot runs keep these base images and labels
        # as a placeholder. They must not be test images, and a new run is never trained.
        train_x = self.to_datums(few_shot["train"], 0, LAST_BASE_LABEL)
        val = self.to_datums(few_shot["val"], 0, LAST_BASE_LABEL) if subsample == "base" else []

        test_items = self.select_test(read_split_file(domainnet_test_path(self.style)))
        test = self.to_datums(test_items, first_label, last_label)
        super().__init__(train_x=train_x, val=val, test=test)

        # DatasetBase reads these from train_x, which would drop any class without train
        # images and, for a new run, give the base class names.
        classnames = self.all_classnames[first_label:last_label + 1]
        self._classnames = classnames
        self._num_classes = len(classnames)
        self._lab2cname = dict(enumerate(classnames))

    def select_test(self, items):
        return items

    def to_datums(self, items, first_label, last_label):
        """Keep the items with labels in the range and renumber them from 0."""
        datums = []
        for item in items:
            label = item["label"]
            if not first_label <= label <= last_label:
                continue
            if item["classname"] != self.all_classnames[label]:
                raise ValueError(
                    "Expected class name '{}' for label {} in {}, found '{}'.".format(
                        self.all_classnames[label], label, item["impath"], item["classname"]
                    )
                )
            datums.append(
                Datum(
                    impath=os.path.join(self.data_root, item["impath"]),
                    label=label - first_label,
                    classname=item["classname"],
                )
            )
        return datums


@DATASET_REGISTRY.register()
class DomainNetB2N_clipart(DomainNetB2NBase):
    style = "clipart"


@DATASET_REGISTRY.register()
class DomainNetB2N_infograph(DomainNetB2NBase):
    style = "infograph"


@DATASET_REGISTRY.register()
class DomainNetB2N_painting(DomainNetB2NBase):
    style = "painting"


@DATASET_REGISTRY.register()
class DomainNetB2N_quickdraw(DomainNetB2NBase):
    style = "quickdraw"


@DATASET_REGISTRY.register()
class DomainNetB2N_real(DomainNetB2NBase):
    style = "real"


@DATASET_REGISTRY.register()
class DomainNetB2N_sketch(DomainNetB2NBase):
    style = "sketch"


@DATASET_REGISTRY.register()
class DomainNetB2NSmoke_real(DomainNetB2N_real):
    """Smoke test only: a few test images per class, to fit the test's 10 minute limit.

    A separate class, so no setting can shrink the test set of a real run.
    """

    def select_test(self, items):
        seen = Counter()
        kept = []
        for item in items:
            seen[item["label"]] += 1
            if seen[item["label"]] <= SMOKE_TEST_IMAGES_PER_CLASS:
                kept.append(item)
        return kept
