import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.io import loadmat

import events_tfds  # ensure url_checksums_dir added

CITATION = """\
@inproceedings{bi2019graph,
title={Graph-based Object Classification for Neuromorphic Vision Sensing},
author={Bi, Y and Chadha, A and Abbas, A and and Bourtsoulatze, E and Andreopoulos, Y},
booktitle={2019 IEEE International Conference on Computer Vision (ICCV)},
year={2019},
organization={IEEE}
}"""

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


def load_event_data(fp):
    data = loadmat(fp)
    time, x, y, polarity = (
        np.squeeze(data[k], axis=-1) for k in ("ts", "x", "y", "pol")
    )
    coords = np.stack((x, y), axis=-1)
    return time, coords, polarity


class AslDvs(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version("0.0.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Event streams for American sign language letters.",
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(  # tfds.features.Sequence(
                        dict(
                            time=tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                            coords=tfds.features.Tensor(
                                shape=(None, 2,), dtype=tf.int64,
                            ),
                            polarity=tfds.features.Tensor(shape=(None,), dtype=tf.bool),
                        )
                    ),
                    "label": tfds.features.ClassLabel(names=CLASSES),
                    "example_id": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download the full MNIST Database
        root_dir = dl_manager.download_and_extract(DL_URL)

        paths = {k: os.path.join(root_dir, f"Yin Bi - {k}.zip") for k in CLASSES}
        for v in paths.values():
            assert tf.io.gfile.exists(v)
        # archives = {k: dl_manager.iter_archive(v) for k, v in paths.items()}
        # scipy.io.loadmat requires seek, so we need to extract these archives.
        archives = dl_manager.extract(paths)
        archives = {k: os.path.join(v, k) for k, v in archives.items()}
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN, gen_kwargs=dict(archives=archives)
            )
        ]

    def _generate_examples(self, archives):
        """Generate NMNIST examples as dicts."""
        for label, archive in archives.items():
            # for path, fp in archive:
            for filename in os.listdir(archive):
                fp = os.path.join(archive, filename)
                example_id = int(filename[-8:-4])
                time, coords, polarity = load_event_data(fp)
                features = dict(
                    events=dict(
                        time=time.astype(np.int64),
                        coords=coords.astype(np.int64),
                        polarity=polarity.astype(bool),
                    ),
                    label=label,
                    example_id=example_id,
                )
                yield (label, example_id), features


if __name__ == "__main__":
    # from scipy.io import loadmat
    # path = '/home/jackd/Downloads/y_4200.mat'
    # loadmat(path)
    download_config = None
    # download_config = tfds.core.download.DownloadConfig(register_checksums=True)
    builder = AslDvs()
    builder.download_and_prepare(download_config=download_config)

    from events_tfds.vis.image import as_frames
    from events_tfds.vis.anim import animate_frames

    for events, labels in builder.as_dataset(split="train", as_supervised=True):
        frames = as_frames(
            **{k: v.numpy() for k, v in events.items()},
            num_frames=24,
            flip_up_down=True,
        )
        print(tf.reduce_max(events["coords"], axis=0).numpy() + 1)
        t = events["time"].numpy()
        print(f"{t.size} events over dt = {t[-1] - t[0]}")
        print(f"class = {CLASSES[labels.numpy()]}")
        animate_frames(frames, fps=8)
