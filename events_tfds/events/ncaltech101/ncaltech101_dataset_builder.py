"""ncaltech101 dataset."""
import typing as tp
import os

import numpy as np
import tensorflow_datasets as tfds

from events_tfds.data_io.neuro import load_neuro_events


HOMEPAGE = "https://www.garrickorchard.com/datasets/n-caltech101"
# mendeley data
DL_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/"
    "cy6cvx3ryv-1.zip"
)
NUM_CLASSES = 102
NAMES_FILE = os.path.join(os.path.dirname(__file__), "caltech101_labels.txt")

GRID_SHAPE = (234, 174)


def load_class_names() -> tp.List[str]:
    with open(NAMES_FILE, encoding="utf-8") as fp:
        return [line for line in fp.readlines() if line]


class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.0.2")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Jitter-event conversion of Caltech101 dataset."),
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(  # tfds.features.Sequence(
                        {
                            "time": tfds.features.Tensor(shape=(None,), dtype=np.int64),
                            "coords": tfds.features.Tensor(
                                shape=(None, 2), dtype=np.int64
                            ),
                            "polarity": tfds.features.Tensor(
                                shape=(None,), dtype=np.bool_
                            ),
                        }
                    ),
                    "label": tfds.features.ClassLabel(names_file=NAMES_FILE),
                    "example_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download the relevant data
        data_folder = dl_manager.download_and_extract(DL_URL)
        (subdir,) = os.listdir(data_folder)
        data_folder = os.path.join(data_folder, subdir)
        zip_path = os.path.join(data_folder, "Caltech101.zip")

        # we provide a single TRAIN split
        # for separate train/validation/test splits, use the tfds.Split API
        # https://www.tensorflow.org/datasets/splits
        return {
            "train": self._generate_examples(archive=dl_manager.iter_archive(zip_path))
        }

    def _generate_examples(self, archive):
        """Generate NMNIST examples as dicts."""
        for path, fobj in archive:
            if not path.endswith(".bin"):
                continue
            _, label, filename = path.split("/")
            example_id = int(filename[6:-4])
            time, coords, polarity = load_neuro_events(fobj)
            features = {
                "events": {
                    "time": time,
                    "coords": coords,
                    "polarity": polarity,
                },
                "label": label.lower(),
                "example_id": example_id,
            }
            yield path, features
