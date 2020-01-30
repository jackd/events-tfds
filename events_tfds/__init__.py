import tensorflow_datasets as tfds
import os
checksums_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), 'url_checksums'))

tfds.core.download.add_checksums_dir(checksums_dir)
