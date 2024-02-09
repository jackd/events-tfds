"""cifar10_dvs dataset."""
import os
import numpy as np
import tensorflow_datasets as tfds

from events_tfds.data_io import dvs

HOMEPAGE = "https://figshare.com/articles/CIFAR10-DVS_New/4724671/2"
DL_URL = "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABrCc6FewNZSNsoObWJqY74a?dl=1"

NUM_CLASSES = 10
GRID_SHAPE = (128, 128)

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "ship",
    "truck",
    "frog",
    "horse",
)

assert len(CLASSES) == NUM_CLASSES


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cifar10_dvs dataset."""

    VERSION = tfds.core.Version("0.0.3")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Jitter-event conversion of CIFAR10 handwritten digits."),
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(
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
        folders = dl_manager.download_and_extract(
            {
                "airplane": "https://ndownloader.figshare.com/files/7712788",
                "automobile": "https://ndownloader.figshare.com/files/7712791",
                "bird": "https://ndownloader.figshare.com/files/7712794",
                "cat": "https://ndownloader.figshare.com/files/7712812",
                "deer": "https://ndownloader.figshare.com/files/7712815",
                "dog": "https://ndownloader.figshare.com/files/7712818",
                "ship": "https://ndownloader.figshare.com/files/7712836",
                "truck": "https://ndownloader.figshare.com/files/7712839",
                "frog": "https://ndownloader.figshare.com/files/7712842",
                "horse": "https://ndownloader.figshare.com/files/7712851",
            }
        )

        return {
            "train": self._generate_examples(
                folders={k: os.path.join(folders[k], k) for k in folders}
            )
        }

    def _generate_examples(self, folders):
        for label, folder in folders.items():
            filenames = os.listdir(folder)
            for filename in filenames:
                assert filename.endswith(".aedat")
                path = os.path.join(folder, filename)
                example_id = int(filename.split("_")[-1][:-6])
                with open(path, "rb") as fp:
                    time, x, y, polarity = dvs.load_events(fp)
                    x = 127 - x
                    polarity = np.logical_not(polarity)
                coords = np.stack((x, y), axis=-1)
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
