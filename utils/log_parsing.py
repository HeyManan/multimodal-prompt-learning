"""Read test accuracy from the log.txt that Dassl writes."""
import re

# Dassl prints accuracy to 1 decimal, and tables need 2, so it is computed from the counts.
TOTAL_LINE = re.compile(r"^\* total: ([\d,]+)$", re.MULTILINE)
CORRECT_LINE = re.compile(r"^\* correct: ([\d,]+)$", re.MULTILINE)


def read_single_count(pattern, text, log_path):
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one line matching '{}' in {}, found {}.".format(
                pattern.pattern, log_path, len(matches)
            )
        )
    return int(matches[0].replace(",", ""))


def parse_counts(log_path):
    """Return (correct, total) from a log that holds exactly one test result."""
    text = log_path.read_text()
    correct = read_single_count(CORRECT_LINE, text, log_path)
    total = read_single_count(TOTAL_LINE, text, log_path)
    return correct, total
