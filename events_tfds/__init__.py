import os

import tensorflow_datasets as tfds

from events_tfds import data_io

CHECKSUMS_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "url_checksums")
)

tfds.core.download.add_checksums_dir(CHECKSUMS_DIR)


# from events_tfds import events

__all__ = [
    "CHECKSUMS_DIR",
    "data_io",
    # 'events',
]
