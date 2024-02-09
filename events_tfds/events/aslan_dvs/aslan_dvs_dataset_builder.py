"""hmdb_dvs dataset."""
import typing as tp
from pathlib import Path
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from scipy.io import loadmat

GRID_SHAPE = (128, 128)


def load_event_data(path: str):
    """Load (time, coords, polarity) from .mat path."""
    data = loadmat(path)
    time, x, y, polarity = (
        np.squeeze(data[k], axis=-1) for k in ("ts", "x", "y", "pol")
    )
    coords = np.stack((x, y), axis=-1)
    return time, coords, polarity


CLASSES = (
    # TODO
)

NUM_CLASSES = len(CLASSES)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for hmdb_dvs dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
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
            homepage="https://github.com/PIX2NVS/NVS2Graph",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        root_dirs = dl_manager.download_and_extract(
            (
                "https://www.dropbox.com/sh/ie75dn246cacf6n/AABgb_yqjuNk3dFTDALhBVQ6a/"
                "Yin%20Bi%20-%20ASLAN-DVS-Part1.zip?dl=1",
                "https://www.dropbox.com/sh/ie75dn246cacf6n/AACokZ6g-mWNBp8t0bxKvlMea/"
                "Yin%20Bi%20-%20ASLAN-DVS-Part2.zip?dl=1",
            )
        )
        raise Exception(root_dirs)
        return {
            "train": self._generate_examples(root_dirs),
        }

    def _generate_examples(self, root_dirs: tp.Iterable[Path]):
        """Yields examples."""
        for root_dir in root_dirs:
            for label in tf.io.gfile.listdir(root_dir):
                for filename in tf.io.gfile.listdir(root_dir / label):
                    path = root_dir / label / filename
                    example_id = filename.split("_")[-1][:-4]
                    try:
                        example_id = int(example_id)
                    except:
                        raise ValueError(
                            f"Invalid example_id '{example_id}' in folder "
                            f"{root_dir / label}"
                        )
                    time, coords, polarity = load_event_data(path)
                    features = {
                        "events": {
                            "time": time.astype(np.int64),
                            "coords": coords.astype(np.int64),
                            "polarity": polarity.astype(bool),
                        },
                        "label": label,
                        "example_id": example_id,
                    }
                    yield filename, features
