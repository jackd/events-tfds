"""poker_dvs dataset."""

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from events_tfds.data_io import dvs

HOMEPAGE = "http://www2.imse-cnm.csic.es/caviar/POKERDVS.html"
DL_URL = "http://www2.imse-cnm.csic.es/caviar/POKER_DVS/poker_dvs.tar.gz"

CLASSES = (
    "spade",
    "club",
    "diamond",
    "heart",
)
NUM_CLASSES = len(CLASSES)
GRID_SHAPE = (33, 35)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for poker_dvs dataset."""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="DVS-events generated from playing car symbols.",
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(  # tfds.features.Sequence(
                        {
                            "time": tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                            "coords": tfds.features.Tensor(
                                shape=(None, 2),
                                dtype=tf.int64,
                            ),
                            "polarity": tfds.features.Tensor(
                                shape=(None,), dtype=tf.bool
                            ),
                        }
                    ),
                    "label": tfds.features.ClassLabel(names=CLASSES),
                    "example_id": tfds.features.Tensor(shape=(), dtype=tf.int64),
                    "inverted": tfds.features.Tensor(shape=(), dtype=tf.bool),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        folder = dl_manager.download_and_extract(DL_URL)

        return {"train": self._generate_examples(folder)}

    def _generate_examples(self, folder):
        """Generate NMNIST examples as dicts."""
        for filename in os.listdir(folder):
            assert filename.endswith(".aedat")
            head = filename[1:-6]
            inverted = head[-1] == "i"
            if inverted:
                head = head[:-1]
            example_id = int(head[-2:])
            label = head[:-2]
            with open(os.path.join(folder, filename), "rb") as fp:
                time, x, y, polarity = dvs.load_events(fp)
                coords = np.stack((x, y), axis=-1)
            features = {
                "events": {
                    "time": time.astype(np.int64),
                    "coords": coords.astype(np.int64),
                    "polarity": polarity,
                },
                "label": label,
                "example_id": example_id,
                "inverted": inverted,
            }
            yield os.path.join(label, str(example_id)), features
