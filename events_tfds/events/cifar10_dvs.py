import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from events_tfds.data_io import dvs

CITATION = """\
@article{chang128,
  title={A 128 128 1.5% Contrast Sensitivity 0.9% FPN 3 $\\mu$s Latency 4 mW Asynchronous Frame-Free Dynamic Vision Sensor Using Transimpedance Preamplifiers, T. Serrano-Gotarredona and B. Linares-Barranco 827 A 3 Megapixel 100 Fps 2.8 m Pixel Pitch CMOS Image Sensor Layer With Built-in Self-Test for 3D Integrated},
  author={Chang, MF and Shen, SJ and Liu, CC and Wu, CW and Lin, YF and King, YC and Lin, CJ and Liao, HJ and Chih, YD and Yamauchi, H}
}"""

HOMEPAGE = 'https://figshare.com/articles/CIFAR10-DVS_New/4724671/2'
DL_URL = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABrCc6FewNZSNsoObWJqY74a?dl=1'

NUM_CLASSES = 10
GRID_SHAPE = (128, 128)

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
                    time, x, y, polarity = dvs.load_events(fp)
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
    #     time, x, y, polarity = dvs.load_events(fp)
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
        print(tf.reduce_max(events['coords'], axis=0).numpy())
        animate_frames(frames, fps=4)
