import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import events_tfds  # ensures checksums directory is added

CITATION = """\
@article{anumula2018feature,
  title={Feature representations for neuromorphic audio spike streams},
  author={Anumula, Jithendar and Neil, Daniel and Delbruck, Tobi and Liu, Shih-Chii},
  journal={Frontiers in neuroscience},
  volume={12},
  pages={23},
  year={2018},
  publisher={Frontiers}
}"""

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
        name = self.name
        super(NtidigitsConfig, self).__init__(
            name=name,
            version=tfds.core.Version("0.0.1"),
            description="neuro-morphic conversion of tigits spoken dataset ({})".format(
                name
            ),
        )

    @property
    def single(self):
        return self._single

    @property
    def name(self):
        return "single" if self.single else "multi"


SINGLE_CONFIG = NtidigitsConfig(single=True)
MULTI_CONFIG = NtidigitsConfig(single=False)


class Ntidigits(tfds.core.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [SINGLE_CONFIG, MULTI_CONFIG]

    def _info(self):
        label = tfds.features.ClassLabel(names=CLASSES)
        if not self.builder_config.single:
            label = tfds.features.Sequence(label)
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(
                        dict(
                            time=tfds.features.Tensor(
                                shape=(None,), dtype=tf.float32
                            ),  # sec
                            channel=tfds.features.Tensor(
                                shape=(None,), dtype=tf.uint8
                            ),  # [0, 64)
                        )
                    ),
                    "label": label,
                    "gender": tfds.features.ClassLabel(names=GENDERS),
                    "speaker_id": tfds.features.Text(),
                    "sample": tfds.features.ClassLabel(names=SAMPLES),
                }
            ),
            supervised_keys=("events", "label"),
            homepage="https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/",
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download(
            "https://www.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=1"
        )
        return [
            tfds.core.SplitGenerator(
                name=split, gen_kwargs=dict(path=path, split=split)
            )
            for split in ("train", "test")
        ]

    def _generate_examples(self, path, split):
        single = self.builder_config.single

        with tf.io.gfile.GFile(path, "rb") as fp:
            root = h5py.File(fp, "r")
            addresses = root["{}_addresses".format(split)]
            timestamps = root["{}_timestamps".format(split)]
            labels = root["{}_labels".format(split)][:]
            for label_str in labels:
                label_str = label_str.decode()
                gender, speaker_id, sample, label = label_str.split("-")
                if single:
                    if len(label) > 1:
                        continue
                else:
                    label = list(label)

                events = dict(
                    time=timestamps[label_str][:],
                    channel=addresses[label_str][:].astype(np.uint8),
                )
                yield label_str, dict(
                    gender=gender,
                    speaker_id=speaker_id,
                    sample=sample,
                    label=label,
                    events=events,
                )


if __name__ == "__main__":
    dl_config = None
    # dl_config = tfds.core.download.DownloadConfig(register_checksums=True)

    Ntidigits(config=SINGLE_CONFIG).download_and_prepare(download_config=dl_config)
