"""mnist_flash_dvs dataset."""

import os
import tensorflow as tf
import tensorflow_datasets as tfds

from events_tfds.data_io.dvs import load_events


HOMEPAGE = "http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html"

NUM_CLASSES = 10


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mnist_flash_dvs dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="DVS-events generated from MNIST handwritten digits.",
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(  # tfds.features.Sequence(
                        {
                            "time": tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                            "coords": tfds.features.Tensor(
                                shape=(None, 2),
                                dtype=tf.int64,
                            ),
                            "polarity": tfds.features.Tensor(
                                shape=(None,), dtype=tf.bool
                            ),
                        }
                    ),
                    "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
                    "example_id": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("events", "label"),
            homepage=HOMEPAGE,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # dl_url = (
        #     "http://www2.imse-cnm.csic.es/caviar/MNIST_FLASH_DVS/"
        #     "Recordings_aedat_zip/Recordings_aedat.zip.{:03d}"
        # )  # these don't work?
        train_url = (
            "http://www2.imse-cnm.csic.es/caviar/MNIST_FLASH_DVS/"
            "Recordings_mat/FMNIST_Train{}.mat"
        )
        test_url = (
            "http://www2.imse-cnm.csic.es/caviar/MNIST_FLASH_DVS/"
            "Recordings_mat/FMNIST_Test.mat"
        )

        folders = dl_manager.download_and_extract(
            {"train": [train_url.format(i) for i in range(1, 7)], "test": [test_url]}
        )
        return {k: self._generate_examples(v) for k, v in folders.items()}

    def _generate_examples(self, paths):
        """Yields examples."""
        raise NotImplementedError("TODO")
        # for path, fobj in archive:
        #     if not path.endswith('.bin'):
        #         continue
        #     _, label, filename = path.split('/')
        #     example_id = int(filename[:-4])
        #     time, x, y, polarity = dvs.load_events(fobj)
        #     coords = np.stack((x, y), axis=-1)
        #     features = dict(events=dict(time=time.astype(np.int64),
        #                                 coords=coords.astype(np.int64),
        #                                 polarity=polarity),
        #                     label=int(label),
        #                     example_id=example_id)
        #     yield example_id, features
