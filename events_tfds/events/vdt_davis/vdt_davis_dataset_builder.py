"""vdt_davis dataset."""
import typing as tp
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

GRID_SHAPE = (240, 180)


class VdtDavisConfig(tfds.core.BuilderConfig):
    def __init__(self, label_frequency: int):
        self.label_frequency = label_frequency
        super().__init__(
            name=f"{label_frequency}hz",
            version=tfds.core.Version("1.0.0"),
            description=f"label frequency {label_frequency}",
        )


def load_events(path: Path) -> tp.Mapping[str, np.ndarray]:
    data = np.asarray(pd.read_csv(path, sep=","), dtype=np.int64)
    return {
        "time": data[:, 0],
        "coords": data[:, 1:3],
        "polarity": data[:, 3].astype(bool),
    }


def load_frames(path: Path):
    filenames = [fn for fn in tf.io.gfile.listdir(path) if fn.endswith(".png")]
    filenames.sort()
    return [{"time": int(fn[:-4]), "image": path / fn} for fn in filenames]


def load_labels(path: Path):
    labels = []
    x_scale, y_scale = GRID_SHAPE
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp.readlines():
            args = line.split(", ")
            time = args[1]
            for i in range(2, len(args), 5):
                left, top, width, height = (float(i) for i in args[i + 1 : i + 5])
                ymin = top / y_scale
                ymax = (top + height) / y_scale
                xmin = left / x_scale
                xmax = (left + width) / x_scale
                if ymax > 1 and ymax < 1.00001:
                    ymax = 1.0
                if xmax > 1 and xmax < 1.00001:
                    xmax = 1.0
                labels.append(
                    {
                        "object_id": int(args[i]),
                        "bounding_box": tfds.features.BBox(ymin, xmin, ymax, xmax),
                        "time": time,
                    }
                )
    return labels


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for vdt_davis dataset."""

    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    BUILDER_CONFIGS = [
        VdtDavisConfig(24),
        VdtDavisConfig(48),
        VdtDavisConfig(96),
        VdtDavisConfig(192),
        VdtDavisConfig(384),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "frames": tfds.features.Sequence(
                        tfds.features.FeaturesDict(
                            {
                                "image": tfds.features.Image(
                                    shape=(*GRID_SHAPE[-1::-1], 1)
                                ),
                                "time": tfds.features.Tensor(shape=(), dtype=tf.int64),
                            }
                        )
                    ),
                    "events": tfds.features.FeaturesDict(
                        {
                            "time": tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                            "coords": tfds.features.Tensor(
                                shape=(None, 2), dtype=tf.int64
                            ),
                            "polarity": tfds.features.Tensor(
                                shape=(None,), dtype=tf.bool
                            ),
                        }
                    ),
                    "labels": tfds.features.Sequence(
                        tfds.features.FeaturesDict(
                            {
                                "object_id": tfds.features.Tensor(
                                    shape=(), dtype=tf.int64
                                ),
                                "bounding_box": tfds.features.BBoxFeature(),
                                "time": tfds.features.Tensor(shape=(), dtype=tf.int64),
                            }
                        )
                    ),
                }
            ),
            supervised_keys=(("frames", "events"), "labels"),
            homepage="https://dataset-homepage/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            "https://drive.google.com/uc?export=download&id="
            "1VO7jvBLGpqbdB1_IZ77kp70zg2atuSax"
        )
        return {
            "Scene_A": self._generate_examples(path / "Dataset", "Scene_A"),
            "Scene_B": self._generate_examples(path / "Dataset", "Scene_B"),
        }

    def _generate_examples(self, path: Path, scene: str):
        """Yields examples."""
        sequences_dir = path / "Traffic Sequences" / scene
        gt_dir = path / "gt" / scene
        sequences = tf.io.gfile.listdir(sequences_dir)
        freq = self.builder_config.label_frequency
        for sequence in sequences:
            sequence_dir = sequences_dir / sequence
            labels_path = f"{sequence}-{'custom' if freq == 24 else 'intp'}{freq}.txt"
            yield sequence, {
                "events": load_events(sequence_dir / f"{sequence}_events.csv"),
                "frames": load_frames(sequence_dir),
                "labels": load_labels(gt_dir / sequence / labels_path),
            }
