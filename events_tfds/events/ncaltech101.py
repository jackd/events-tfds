import os

import tensorflow as tf
import tensorflow_datasets as tfds

from events_tfds.data_io.neuro import load_neuro_events

CITATION = """\
@article{orchard2015converting,
  title={Converting static image datasets to spiking neuromorphic datasets using saccades},
  author={Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K and Thakor, Nitish},
  journal={Frontiers in neuroscience},
  volume={9},
  pages={437},
  year={2015},
  publisher={Frontiers}
}"""

HOMEPAGE = "https://www.garrickorchard.com/datasets/n-caltech101"
DL_URL = "https://www.dropbox.com/sh/iuv7o3h2gv6g4vd/AADYPdhIBK7g_fPCLKmG6aVpa?dl=1"
NUM_CLASSES = 102
NAMES_FILE = os.path.join(os.path.dirname(__file__), "caltech101_labels.txt")

GRID_SHAPE = (234, 174)


class Ncaltech101(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version("0.0.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Jitter-event conversion of Caltech101 dataset."),
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(  # tfds.features.Sequence(
                        dict(
                            time=tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                            coords=tfds.features.Tensor(
                                shape=(None, 2), dtype=tf.int64
                            ),
                            polarity=tfds.features.Tensor(shape=(None,), dtype=tf.bool),
                        )
                    ),
                    "label": tfds.features.ClassLabel(names_file=NAMES_FILE),
                    "example_id": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download the relevant data
        data_folder = dl_manager.download_and_extract(DL_URL)
        zip_path = os.path.join(data_folder, "Caltech101.zip")

        # we provide a single TRAIN split
        # for separate train/validation/test splits, use the tfds.Split API
        # https://www.tensorflow.org/datasets/splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(archive=dl_manager.iter_archive(zip_path)),
            )
        ]

    def _generate_examples(self, archive):
        """Generate NMNIST examples as dicts."""
        for path, fobj in archive:
            if not path.endswith(".bin"):
                continue
            _, label, filename = path.split("/")
            example_id = int(filename[6:-4])
            time, coords, polarity = load_neuro_events(fobj)
            features = dict(
                events=dict(time=time, coords=coords, polarity=polarity),
                label=label.lower(),
                example_id=example_id,
            )
            yield (label, example_id), features


if __name__ == "__main__":
    from events_tfds.vis.image import as_frames
    from events_tfds.vis.anim import animate_frames

    dl_config = None
    # download_config=tfds.core.download.DownloadConfig(
    #         register_checksums=True)
    builder = Ncaltech101()
    builder.download_and_prepare(download_config=dl_config)
    splits = builder.info.splits
    class_names = builder.info.features["label"].names

    for k in sorted(splits):
        print(f"{k}: {splits[k].num_examples}")

    for events, labels in builder.as_dataset(split="train", as_supervised=True):
        frames = as_frames(**{k: v.numpy() for k, v in events.items()}, num_frames=20)
        t = events["time"].numpy()
        print(f"{t.size} events over {t[-1] - t[0]} dt")
        print(class_names[labels.numpy()])
        print(tf.reduce_max(events["coords"], axis=0).numpy() + 1)
        animate_frames(frames, fps=4)
