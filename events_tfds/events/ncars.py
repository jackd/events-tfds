import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import events_tfds  # ensures checksums directory is added

CITATION = """\
@inproceedings{sironi2018hats,
  title={HATS: Histograms of averaged time surfaces for robust event-based object classification},
  author={Sironi, Amos and Brambilla, Manuele and Bourdis, Nicolas and Lagorce, Xavier and Benosman, Ryad},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1731--1740},
  year={2018}
}"""

HOMEPAGE = 'https://www.prophesee.ai/dataset-n-cars-download/'
DL_URL = 'http://www.prophesee.ai/resources/Prophesee_Dataset_n_cars.zip'

BACKGROUND = 'background'
CAR = 'car'

CLASSES = (BACKGROUND, CAR)
NUM_CLASSES = 2


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


def load_atis_events(fp):
    # strip header
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == '%':
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    fp.seek(p + 2)
    data = np.fromstring(fp.read(), dtype='<u4')

    time = data[::2]
    coords = data[1::2]

    x = read_bits(coords, x_mask, x_shift)
    y = read_bits(coords, y_mask, y_shift)
    pol = read_bits(coords, pol_mask, pol_shift)
    coords = np.stack((x, y), axis=-1)
    return time, coords, pol.astype(np.bool)


class Ncars(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version("0.0.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Binary car classification problem.",
            features=tfds.features.FeaturesDict({
                "events":  # tfds.features.Sequence(
                    tfds.features.FeaturesDict(
                        dict(time=tfds.features.Tensor(shape=(None,),
                                                       dtype=tf.int64),
                             coords=tfds.features.Tensor(shape=(None, 2),
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
        data_folder = dl_manager.download_and_extract(DL_URL)
        test_dir = dl_manager.extract(
            os.path.join(data_folder, 'n-cars_test.zip'))
        train_dir = dl_manager.extract(
            os.path.join(data_folder, 'n-cars_train.zip'))

        # MNIST provides TRAIN and TEST splits, not a VALIDATION split, so we only
        # write the TRAIN and TEST splits to disk.
        return [
            tfds.core.SplitGenerator(name=tfds.Split.TEST,
                                     gen_kwargs=dict(root_dir=test_dir)),
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                     gen_kwargs=dict(root_dir=train_dir)),
        ]

    def _generate_examples(self, root_dir):
        """Generate NMNIST examples as dicts."""
        for folder, _, filenames in os.walk(root_dir):
            if len(filenames) > 0:
                label = os.path.split(folder)[1]
                if label == 'cars':
                    label = CAR
                for filename in filenames:
                    if not filename.endswith('.dat'):
                        continue

                    example_id = int(filename[4:-7])
                    with tf.io.gfile.GFile(os.path.join(folder, filename),
                                           'rb') as fobj:
                        time, coords, polarity = load_atis_events(fobj)
                    features = dict(events=dict(time=time.astype(np.int64),
                                                coords=coords.astype(np.int64),
                                                polarity=polarity),
                                    label=label,
                                    example_id=example_id)
                    yield (label, example_id), features


if __name__ == '__main__':
    download_config = None
    # download_config = tfds.core.download.DownloadConfig(
    #     register_checksums=True)
    Ncars().download_and_prepare(download_config=download_config)

    # from events_tfds.vis.anim import animate_frames
    import matplotlib.pyplot as plt
    from events_tfds.vis.image import as_frame

    for events, labels in tfds.load('ncars', split='train', as_supervised=True):
        print(CLASSES[labels.numpy()])
        coords = events['coords'].numpy()
        print(len(coords))
        print(np.min(coords, axis=0), np.max(coords, axis=0) + 1)
        t = events['time'].numpy()
        print('t extends: ', np.min(t), np.max(t))
        frame = as_frame(
            coords=coords,
            # time=events['time'],
            polarity=events['polarity'].numpy(),
        )
        plt.imshow(frame)
        plt.show()
