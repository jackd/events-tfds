"""nmnist dataset."""

import os

import tensorflow as tf
import tensorflow_datasets as tfds

from events_tfds.data_io.neuro import load_neuro_events


HOMEPAGE = "https://www.garrickorchard.com/datasets/n-mnist"
# mendeley
DL_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/"
    "468j46mzdv-1.zip"
)
NUM_CLASSES = 10

GRID_SHAPE = (34, 34)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for nmnist dataset."""

    VERSION = tfds.core.Version("0.0.2")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Jitter-event conversion of MNIST handwritten digits."),
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(
                        {
                            "time": tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                            "coords": tfds.features.Tensor(
                                shape=(None, 2), dtype=tf.int64
                            ),
                            "polarity": tfds.features.Tensor(
                                shape=(None,), dtype=tf.bool
                            ),
                        }
                    ),
                    "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
                    "example_id": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download the full MNIST Database
        data_folder = dl_manager.download_and_extract(DL_URL)
        (subdir,) = os.listdir(data_folder)
        data_folder = os.path.join(data_folder, subdir)

        # MNIST provides TRAIN and TEST splits, not a VALIDATION split, so we only
        # write the TRAIN and TEST splits to disk.
        return {
            "train": self._generate_examples(
                archive=dl_manager.iter_archive(os.path.join(data_folder, "Train.zip"))
            ),
            "test": self._generate_examples(
                archive=dl_manager.iter_archive(os.path.join(data_folder, "Test.zip"))
            ),
        }

    def _generate_examples(self, archive):
        """Generate NMNIST examples as dicts."""
        for path, fobj in archive:
            if not path.endswith(".bin"):
                continue
            _, label, filename = path.split("/")
            example_id = int(filename[:-4])
            time, coords, polarity = load_neuro_events(fobj)
            features = {
                "events": {
                    "time": time,
                    "coords": coords,
                    "polarity": polarity,
                },
                "label": int(label),
                "example_id": example_id,
            }
            yield example_id, features
