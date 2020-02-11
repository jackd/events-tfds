import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import events_tfds.data_io.aedat as aedat

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

HOMEPAGE = 'https://figshare.com/articles/CIFAR10-DVS_New/4724671/2'
DL_URL = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABrCc6FewNZSNsoObWJqY74a?dl=1'

NUM_CLASSES = 10

CLASSES = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'ship',
    'truck',
    'frog',
    'horse',
)

assert (len(CLASSES) == NUM_CLASSES)


def load_events(fp):
    return aedat.load_events(fp,
                             bytes_trim=0,
                             bytes_skip=0,
                             x_mask=0xfE,
                             x_shift=1,
                             y_mask=0x7f00,
                             y_shift=8,
                             polarity_mask=1,
                             polarity_shift=None,
                             times_first=False)


class Cifar10DVS(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version("0.0.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Jitter-event conversion of MNIST handwritten digits."),
            features=tfds.features.FeaturesDict({
                "events":  # tfds.features.Sequence(
                    tfds.features.FeaturesDict(
                        dict(time=tfds.features.Tensor(shape=(None,),
                                                       dtype=tf.int64),
                             coords=tfds.features.Tensor(shape=(
                                 None,
                                 2,
                             ),
                                                         dtype=tf.int64),
                             polarity=tfds.features.Tensor(shape=(None,),
                                                           dtype=tf.bool))),
                "label":
                    tfds.features.ClassLabel(names=CLASSES),
                "example_id":
                    tfds.features.Tensor(shape=(), dtype=tf.int64)
            }),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download the full MNIST Database
        folders = dl_manager.download_and_extract({
            'airplane': 'https://ndownloader.figshare.com/files/7712788',
            'automobile': 'https://ndownloader.figshare.com/files/7712791',
            'bird': 'https://ndownloader.figshare.com/files/7712794',
            'cat': 'https://ndownloader.figshare.com/files/7712812',
            'deer': 'https://ndownloader.figshare.com/files/7712815',
            'dog': 'https://ndownloader.figshare.com/files/7712818',
            'ship': 'https://ndownloader.figshare.com/files/7712836',
            'truck': 'https://ndownloader.figshare.com/files/7712839',
            'frog': 'https://ndownloader.figshare.com/files/7712842',
            'horse': 'https://ndownloader.figshare.com/files/7712851',
        })

        # MNIST provides TRAIN and TEST splits, not a VALIDATION split, so we only
        # write the TRAIN and TEST splits to disk.
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(
                    folders={k: os.path.join(folders[k], k) for k in folders})),
        ]

    def _generate_examples(self, folders):
        """Generate NMNIST examples as dicts."""
        for label, folder in folders.items():
            filenames = os.listdir(folder)
            for filename in filenames:
                assert (filename.endswith('.aedat'))
                path = os.path.join(folder, filename)
                example_id = int(filename.split('_')[-1][:-6])
                with open(path, 'rb') as fp:
                    time, x, y, polarity = load_events(fp)
                coords = np.stack((x, y), axis=-1)
                features = dict(events=dict(time=time.astype(np.int64),
                                            coords=coords.astype(np.int64),
                                            polarity=polarity.astype(bool)),
                                label=label,
                                example_id=example_id)
                yield (label, example_id), features


if __name__ == '__main__':

    download_config = None
    # download_config = tfds.core.download.DownloadConfig(
    #     register_checksums=True)
    Cifar10DVS().download_and_prepare(download_config=download_config)

    from events_tfds.vis.image import as_frames
    from events_tfds.vis.anim import animate_frames
    # path = '/tmp/cifar10_ship_999.aedat'
    # with open(path, 'rb') as fp:
    #     time, x, y, polarity = load_events(fp)
    # coords = np.stack((x, y), axis=-1)
    # print(time[:100])
    # print(coords.shape)
    # print(polarity.shape)
    # print(time.shape)
    # print(np.max(coords, axis=0))
    # frames = as_frames(time=time,
    #                    coords=coords,
    #                    polarity=polarity,
    #                    num_frames=20)
    # print(len(time))
    # animate_frames(frames, fps=4)
    # exit()

    for events, labels in tfds.load('cifar10_dvs',
                                    split='train',
                                    as_supervised=True):
        frames = as_frames(**{k: v.numpy() for k, v in events.items()},
                           num_frames=20)
        print(labels.numpy())
        animate_frames(frames, fps=4)
