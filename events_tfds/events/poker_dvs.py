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

HOMEPAGE = 'http://www2.imse-cnm.csic.es/caviar/POKERDVS.html'
DL_URL = 'http://www2.imse-cnm.csic.es/caviar/POKER_DVS/poker_dvs.tar.gz'

CLASSES = (
    'spade',
    'club',
    'diamond',
    'heart',
)


class PokerDVS(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version("0.0.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="DVS-events generated from playing car symbols.",
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
                    tfds.features.Tensor(shape=(), dtype=tf.int64),
                "inverted":
                    tfds.features.Tensor(shape=(), dtype=tf.bool),
            }),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        folder = dl_manager.download_and_extract(DL_URL)

        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                     gen_kwargs=dict(folder=folder)),
        ]

    def _generate_examples(self, folder):
        """Generate NMNIST examples as dicts."""
        for filename in os.listdir(folder):
            assert (filename.endswith('.aedat'))
            head = filename[1:-6]
            inverted = head[-1] == 'i'
            if inverted:
                head = head[:-1]
            example_id = int(head[-2:])
            label = head[:-2]
            with open(os.path.join(folder, filename), 'rb') as fp:
                time, x, y, polarity = dvs.load_events(fp)
                coords = np.stack((x, y), axis=-1)
            features = dict(events=dict(time=time.astype(np.int64),
                                        coords=coords.astype(np.int64),
                                        polarity=polarity),
                            label=label,
                            example_id=example_id,
                            inverted=inverted)
            yield (label, example_id), features


if __name__ == '__main__':
    download_config = None
    # download_config = tfds.core.download.DownloadConfig(
    #     register_checksums=True)
    PokerDVS().download_and_prepare(download_config=download_config)

    from events_tfds.vis.image import as_frames
    from events_tfds.vis.anim import animate_frames

    for events, labels in tfds.load('poker_dvs',
                                    split='train',
                                    as_supervised=True):
        frames = as_frames(**{k: v.numpy() for k, v in events.items()},
                           num_frames=20)
        print(labels.numpy())
        animate_frames(frames, fps=4)
