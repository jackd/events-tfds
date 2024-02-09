"""asl_dvs dataset."""
import os

import numpy as np
import tensorflow_datasets as tfds
from scipy.io import loadmat


HOMEPAGE = "https://github.com/PIX2NVS/NVS2Graph"
DL_URL = "https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=1"

NUM_CLASSES = 24
GRID_SHAPE = (240, 180)

CLASSES = (
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
)

assert len(CLASSES) == NUM_CLASSES


def load_event_data(path: str):
    """Load (time, coords, polarity) from .mat path."""
    data = loadmat(path)
    time, x, y, polarity = (
        np.squeeze(data[k], axis=-1) for k in ("ts", "x", "y", "pol")
    )
    coords = np.stack((x, y), axis=-1)
    return time, coords, polarity


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for asl_dvs dataset."""

    VERSION = tfds.core.Version("0.0.2")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Event streams for American sign language letters.",
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(  # tfds.features.Sequence(
                        {
                            "time": tfds.features.Tensor(shape=(None,), dtype=np.int64),
                            "coords": tfds.features.Tensor(
                                shape=(None, 2),
                                dtype=np.int64,
                            ),
                            "polarity": tfds.features.Tensor(
                                shape=(None,), dtype=np.bool_
                            ),
                        }
                    ),
                    "label": tfds.features.ClassLabel(names=CLASSES),
                    "example_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        root_dir = dl_manager.download_and_extract(DL_URL)

        paths = {k: os.path.join(root_dir, f"Yin Bi - {k}.zip") for k in CLASSES}
        archives = dl_manager.extract(paths)
        archives = {k: os.path.join(v, k) for k, v in archives.items()}
        return {"train": self._generate_examples(archives)}

    def _generate_examples(self, archives):
        """Generate examples as dicts."""
        for label, archive in archives.items():
            for filename in os.listdir(archive):
                fp = os.path.join(archive, filename)
                example_id = int(filename[-8:-4])
                time, coords, polarity = load_event_data(fp)
                features = {
                    "events": {
                        "time": time.astype(np.int64),
                        "coords": coords.astype(np.int64),
                        "polarity": polarity.astype(bool),
                    },
                    "label": label,
                    "example_id": example_id,
                }
                yield os.path.join(label, filename), features
