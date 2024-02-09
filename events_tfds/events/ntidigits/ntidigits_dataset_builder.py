"""ntidigits dataset."""
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


CLASSES = (
    "z",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "o",
)

NUM_CLASSES = len(CLASSES)
NUM_CHANNELS = 64

GENDERS = (
    "man",
    "woman",
)

SAMPLES = ("a", "b")


class NtidigitsConfig(tfds.core.BuilderConfig):
    def __init__(self, single=True):
        self._single = single
        name = "single" if self.single else "multi"
        super().__init__(
            name=name,
            version=tfds.core.Version("1.0.0"),
            description=f"neuro-morphic conversion of tigits spoken dataset ({name})",
        )

    @property
    def single(self):
        return self._single


SINGLE_CONFIG = NtidigitsConfig(single=True)
MULTI_CONFIG = NtidigitsConfig(single=False)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ntidigits dataset."""

    BUILDER_CONFIGS = [SINGLE_CONFIG, MULTI_CONFIG]
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        label = tfds.features.ClassLabel(names=CLASSES)
        if not self.builder_config.single:
            label = tfds.features.Sequence(label)
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(
                        {
                            "time": tfds.features.Tensor(
                                shape=(None,), dtype=tf.float32
                            ),  # sec
                            "channel": tfds.features.Tensor(
                                shape=(None,), dtype=tf.uint8
                            ),  # [0, 64)
                        }
                    ),
                    "label": label,
                    "gender": tfds.features.ClassLabel(names=GENDERS),
                    "speaker_id": tfds.features.Text(),
                    "sample": tfds.features.ClassLabel(names=SAMPLES),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=(
                "https://docs.google.com/document/d/"
                "1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/"
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download(
            "https://www.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=1"
        )
        return {
            split: self._generate_examples(path, split) for split in ("train", "test")
        }

    def _generate_examples(self, path, split):
        """Yields examples."""
        single = self.builder_config.single

        with tf.io.gfile.GFile(path, "rb") as fp:
            root = h5py.File(fp, "r")
            addresses = root[f"{split}_addresses"]
            timestamps = root[f"{split}_timestamps"]
            labels = root[f"{split}_labels"][:]
            for label_str in labels:
                label_str = label_str.decode()
                gender, speaker_id, sample, label = label_str.split("-")
                if single:
                    if len(label) > 1:
                        continue
                else:
                    label = list(label)

                yield label_str, {
                    "gender": gender,
                    "speaker_id": speaker_id,
                    "sample": sample,
                    "label": label,
                    "events": {
                        "time": timestamps[label_str][:],
                        "channel": addresses[label_str][:].astype(np.uint8),
                    },
                }
