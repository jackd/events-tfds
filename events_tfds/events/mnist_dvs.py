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

HOMEPAGE = 'http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html'
DL_URL = 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data{}.zip'
NUM_CLASSES = 10


class MnistDvsConfig(tfds.core.BuilderConfig):

    def __init__(self, scale: int, version=tfds.core.Version("0.0.1")):
        self._scale = scale
        super(MnistDvsConfig,
              self).__init__(name='scale{:02d}'.format(scale),
                             version=tfds.core.Version("0.0.1"),
                             description="scale == {}".format(scale))

    @property
    def scale(self):
        return self._scale


SCALE4 = MnistDvsConfig(4)
SCALE8 = MnistDvsConfig(8)
SCALE16 = MnistDvsConfig(16)


class MnistDVS(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [SCALE4, SCALE8, SCALE16]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="DVS-events generated from MNIST handwritten digits.",
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
                    tfds.features.ClassLabel(num_classes=NUM_CLASSES),
                "example_id":
                    tfds.features.Tensor(shape=(), dtype=tf.int64)
            }),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        folders = dl_manager.download_and_extract(
            {i: DL_URL.format(i) for i in range(NUM_CLASSES)})
        folders = tuple(
            os.path.join(folders[i], 'grabbed_data{}'.format(i))
            for i in range(NUM_CLASSES))
        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                     gen_kwargs=dict(
                                         folders=folders,
                                         scale=self.builder_config.scale)),
        ]

    def _generate_examples(self, folders, scale):
        """Generate NMNIST examples as dicts."""
        for folder in folders:
            label = int(folder[-1])
            folder = os.path.join(folder, 'scale{}'.format(scale))
            for filename in os.listdir(folder):
                example_id = int(filename[-10:-6])
                with open(os.path.join(folder, filename), 'rb') as fp:
                    time, x, y, polarity = dvs.load_events(fp)
                    coords = np.stack((x, y), axis=-1)
                features = dict(events=dict(time=time.astype(np.int64),
                                            coords=coords.astype(np.int64),
                                            polarity=polarity),
                                label=label,
                                example_id=example_id)
                yield (label, example_id), features


if __name__ == '__main__':
    from events_tfds.vis.image import as_frames
    from events_tfds.vis.anim import animate_frames
    download_config = None
    # download_config = tfds.core.download.DownloadConfig(register_checksums=True)
    MnistDVS().download_and_prepare(download_config=download_config)

    for events, labels in tfds.load('mnist_dvs',
                                    split='train',
                                    as_supervised=True):
        frames = as_frames(**{k: v.numpy() for k, v in events.items()},
                           num_frames=20)
        print(labels.numpy())
        print(tf.reduce_max(events['coords'], axis=0).numpy())
        animate_frames(frames, fps=4)
