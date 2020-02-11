import tensorflow_datasets as tfds
import os
import collections

Event = collections.namedtuple('Event', ('time', 'coords', 'polarity'))

CHECKSUMS_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), 'url_checksums'))

tfds.core.download.add_checksums_dir(CHECKSUMS_DIR)
