import tensorflow as tf
import tensorflow_datasets as tfds

CITATION = """\
@article{chang128,
  title={A 128 128 1.5% Contrast Sensitivity 0.9% FPN 3 $\\mu$s Latency 4 mW Asynchronous Frame-Free Dynamic Vision Sensor Using Transimpedance Preamplifiers, T. Serrano-Gotarredona and B. Linares-Barranco 827 A 3 Megapixel 100 Fps 2.8 m Pixel Pitch CMOS Image Sensor Layer With Built-in Self-Test for 3D Integrated},
  author={Chang, MF and Shen, SJ and Liu, CC and Wu, CW and Lin, YF and King, YC and Lin, CJ and Liao, HJ and Chih, YD and Yamauchi, H}
}"""

HOMEPAGE = "http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html"

NUM_CLASSES = 10


class MnistFlashDVS(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.0.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="DVS-events generated from MNIST handwritten digits.",
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(  # tfds.features.Sequence(
                        dict(
                            time=tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                            coords=tfds.features.Tensor(
                                shape=(
                                    None,
                                    2,
                                ),
                                dtype=tf.int64,
                            ),
                            polarity=tfds.features.Tensor(shape=(None,), dtype=tf.bool),
                        )
                    ),
                    "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
                    "example_id": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        raise Exception("TODO - issues using downloaded data at the moment...")

        # url = ('http://www2.imse-cnm.csic.es/caviar/MNIST_FLASH_DVS/'
        #        'Recordings_aedat_zip/Recordings_aedat.zip.{:03d}')
        # num_recordings = 14
        # folders = dl_manager.download_and_extract(
        #     {i: url.format(i + 1) for i in range(num_recordings)})
        # folders = tuple(folders[i] for i in range(num_recordings))
        # print('/'.join(folders))

        # # MNIST provides TRAIN and TEST splits, not a VALIDATION split, so we only
        # # write the TRAIN and TEST splits to disk.
        # return [
        #     tfds.core.SplitGenerator(name=tfds.Split.TEST,
        #                              gen_kwargs=dict(folders=folders,
        #                                              split='test')),
        #     tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
        #                              gen_kwargs=dict(folders=folders,
        #                                              split='train')),
        # ]

    def _generate_examples(self, folders, split):
        """Generate NMNIST examples as dicts."""
        raise NotImplementedError("TODO")

    #     for path, fobj in archive:
    #         if not path.endswith('.bin'):
    #             continue
    #         _, label, filename = path.split('/')
    #         example_id = int(filename[:-4])
    #         time, x, y, polarity = dvs.load_events(fobj)
    #         coords = np.stack((x, y), axis=-1)
    #         features = dict(events=dict(time=time.astype(np.int64),
    #                                     coords=coords.astype(np.int64),
    #                                     polarity=polarity),
    #                         label=int(label),
    #                         example_id=example_id)
    #         yield example_id, features


if __name__ == "__main__":
    # from events_tfds.vis.image import as_frames
    # from events_tfds.vis.anim import animate_frames
    # download_config = None
    download_config = tfds.core.download.DownloadConfig(register_checksums=True)
    MnistFlashDVS().download_and_prepare(download_config=download_config)

    # for events, labels in tfds.load('nmnist', split='train',
    #                                 as_supervised=True):
    #     frames = as_frames(**{k: v.numpy() for k, v in events.items()},
    #                        num_frames=20)
    #     print(labels.numpy())
    #     animate_frames(frames, fps=4)
