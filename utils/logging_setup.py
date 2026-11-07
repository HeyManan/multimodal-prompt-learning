"""Logging setup shared by the project's scripts."""
import logging
import sys


def setup_logging():
    logging.basicConfig(
        stream=sys.stdout, force=True, level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    # An import in the Dassl stack disables logging, so turn it back on.
    logging.disable(logging.NOTSET)
