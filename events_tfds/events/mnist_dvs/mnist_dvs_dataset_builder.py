"""mnist_dvs dataset."""

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from events_tfds.data_io import dvs

HOMEPAGE = "http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html"
DL_URL = "http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data{}.zip"
NUM_CLASSES = 10

# all scales are the same shape
GRID_SHAPE = (128, 128)


class MnistDvsConfig(tfds.core.BuilderConfig):
    def __init__(self, scale: int, version=tfds.core.Version("0.0.1")):
        self._scale = scale
        super().__init__(
            name=f"scale{scale:02d}",
            version=version,
            description=f"scale == {scale}",
        )

    @property
    def scale(self):
        return self._scale


SCALE4 = MnistDvsConfig(4)
SCALE8 = MnistDvsConfig(8)
SCALE16 = MnistDvsConfig(16)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mnist_dvs dataset."""

    BUILDER_CONFIGS = [SCALE4, SCALE8, SCALE16]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="DVS-events generated from MNIST handwritten digits.",
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(
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
                    "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
                    "example_id": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        folders = dl_manager.download_and_extract(
            {i: DL_URL.format(i) for i in range(NUM_CLASSES)}
        )
        folders = tuple(
            os.path.join(folders[i], f"grabbed_data{i}") for i in range(NUM_CLASSES)
        )
        return {
            "train": self._generate_examples(
                folders=folders, scale=self.builder_config.scale
            ),
        }

    def _generate_examples(self, folders, scale):
        """Generate NMNIST examples as dicts."""
        for folder in folders:
            label = int(folder[-1])
            folder = os.path.join(folder, f"scale{scale}")
            for filename in os.listdir(folder):
                example_id = int(filename[-10:-6])
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
                }
                yield os.path.join(str(label), str(example_id)), features
