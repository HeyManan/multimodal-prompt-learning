"""Gather every results.json under $OUTPUT/base2new into results/runs.csv."""
import csv
import json
import logging

from utils.logging_setup import setup_logging
from utils.paths import RUNS_FOLDER, require_env_dir
from utils.results_table import RUNS_CSV


def read_records(output_root):
    runs_root = output_root / RUNS_FOLDER
    paths = sorted(runs_root.glob("*/*/*/results.json"))
    if not paths:
        raise FileNotFoundError("Expected results.json files under {}, found none.".format(runs_root))
    return [(path, json.loads(path.read_text())) for path in paths]


def check_records(records):
    expected_fields = set(records[0][1])
    for path, record in records:
        if set(record) != expected_fields:
            raise RuntimeError(
                "Expected fields {} in {}, found {}.".format(
                    sorted(expected_fields), path, sorted(record)
                )
            )
        if record["smoke"]:
            raise RuntimeError("Expected no smoke runs in the results, found {}.".format(path))


def write_csv(records, csv_path):
    fields = list(records[0][1])
    csv_path.parent.mkdir(exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(record for _, record in records)


def main():
    setup_logging()
    records = read_records(require_env_dir("OUTPUT"))
    check_records(records)
    write_csv(records, RUNS_CSV)
    logging.info("Wrote %d runs to %s", len(records), RUNS_CSV)


if __name__ == "__main__":
    main()
