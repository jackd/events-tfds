"""ncars dataset."""
import typing as tp
import os

import numpy as np
import tensorflow_datasets as tfds


HOMEPAGE = "https://www.prophesee.ai/dataset-n-cars-download/"
DL_URL = "http://www.prophesee.ai/resources/Prophesee_Dataset_n_cars.zip"

BACKGROUND = "background"
CAR = "car"

CLASSES = (BACKGROUND, CAR)
NUM_CLASSES = 2

GRID_SHAPE = (100, 120)  # all(?) are smaller, though these are the max values


def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


x_mask = 0x00003FFF
y_mask = 0x0FFFC000
pol_mask = 0x10000000
x_shift = 0
y_shift = 14
pol_shift = 28


def load_atis_events(fp) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # strip header
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "%":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    fp.seek(p + 2)
    data = np.fromstring(fp.read(), dtype="<u4")

    time = data[::2]
    coords = data[1::2]

    x = read_bits(coords, x_mask, x_shift)
    y = read_bits(coords, y_mask, y_shift)
    pol = read_bits(coords, pol_mask, pol_shift)
    coords = np.stack((y, x), axis=-1)
    return time, coords, pol.astype(bool)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ncars dataset."""

    VERSION = tfds.core.Version("1.0.1")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.0.1": "Changed to (y, x) order.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="Binary car classification problem.",
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
                    "label": tfds.features.ClassLabel(names=CLASSES),
                    "example_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_folder = dl_manager.download_and_extract(DL_URL)
        test_dir = dl_manager.extract(os.path.join(data_folder, "n-cars_test.zip"))
        train_dir = dl_manager.extract(os.path.join(data_folder, "n-cars_train.zip"))

        return {
            "train": self._generate_examples(train_dir),
            "test": self._generate_examples(test_dir),
        }

    def _generate_examples(self, root_dir):
        """Yields examples."""
        for folder, _, filenames in os.walk(root_dir):
            if len(filenames) > 0:
                label = os.path.split(folder)[1]
                if label == "cars":
                    label = CAR
                for filename in filenames:
                    if not filename.endswith(".dat"):
                        continue

                    example_id = int(filename[4:-7])
                    with open(os.path.join(folder, filename), "rb") as fobj:
                        time, coords, polarity = load_atis_events(fobj)
                    features = {
                        "events": {
                            "time": time.astype(np.int64),
                            "coords": coords.astype(np.int64),
                            "polarity": polarity,
                        },
                        "label": label,
                        "example_id": example_id,
                    }
                    yield os.path.join(label, str(example_id)), features
