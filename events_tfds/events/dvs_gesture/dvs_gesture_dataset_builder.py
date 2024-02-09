"""dvs_gesture dataset."""

from pathlib import Path
import struct

import tensorflow_datasets as tfds
import pandas as pd
import numpy as np


def skip_comments(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def load_events(fp):
    # aedat3.1
    times = []
    xs = []
    ys = []
    polarities = []
    event_dtype = np.dtype([("data", np.uint32), ("timestamp", np.uint32)])
    p = skip_comments(fp)
    fp.seek(p)
    while True:
        header = fp.read(28)
        if len(header) == 0:
            break
        (
            event_type,
            event_source,
            event_size,
            event_ts_offset,
            event_ts_overflow,
            event_capacity,
            event_number,
            event_valid,
        ) = struct.unpack("HHIIIIII", header)
        del event_type, event_source, event_ts_offset, event_ts_overflow, event_valid
        packet_size = event_size * event_capacity
        events = np.fromstring(fp.read(packet_size), dtype=event_dtype)
        assert events.shape == (event_number,), (events.shape, event_number)
        data = events["data"]
        time = events["timestamp"]

        assert np.all(time[1:] >= time[:-1]), time

        # Extract x, y, and polarity from data using vectorized numpy operations
        x = np.bitwise_and(np.right_shift(data, 17), 0x00001FFF)
        x = 127 - x
        y = np.bitwise_and(np.right_shift(data, 2), 0x00001FFF)
        polarity = np.bitwise_and(np.right_shift(data, 1), 0x00000001)

        times.append(time)
        xs.append(x)
        ys.append(y)
        polarities.append(polarity)
    times = np.concatenate(times)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    polarities = np.concatenate(polarities)
    order = np.argsort(times)
    return times[order], xs[order], ys[order], polarities[order]


CLASSES = (
    "hand clapping",
    "right hand wave",
    "left hand wave",
    "right arm clockwise",
    "right arm counter clockwise",
    "left arm clockwise",
    "left arm counter clockwise",
    "arm roll",
    "air drums",
    "air guitar",
    "other gestures",
)
NUM_CLASSES = len(CLASSES)  # 11
GRID_SHAPE = (128, 128)

FLIP_LR_LABEL_MAP = (
    0,  # hand clapping,
    2,  # right hand wave -> left hand wave
    1,  # left hand wave -> right hand wave
    6,  # right arm clockwise -> left arm counter clockwise
    5,  # right arm counter clockwise -> left arm clockwise
    4,  # left arm clockwise -> right arm counter clockwise
    3,  # left arm counter clockwise -> right arm clockwise
    7,  # arm roll
    8,  # air drums
    9,  # air guitar
    10,  # other gestures
)

FLIP_UD_LABEL_MAP = (
    0,  # hand clapping,
    2,  # right hand wave -> left hand wave
    1,  # left hand wave -> right hand wave
    6,  # right arm clockwise -> left arm counter clockwise
    5,  # right arm counter clockwise -> left arm clockwise
    4,  # left arm clockwise -> right arm counter clockwise
    3,  # left arm counter clockwise -> right arm clockwise
    7,  # arm roll
    8,  # air drums
    9,  # air guitar
    10,  # other gestures
)

FLIP_TIME_LABEL_MAP = (
    0,  # hand clapping,
    1,  # right hand wave
    2,  # left hand wave
    4,  # right arm clockwise -> right arm counter clockwise
    3,  # right arm counter clockwise -> right arm clockwise
    6,  # left arm clockwise -> left arm counter clockwise
    5,  # left arm counter clockwise -> left arm clockwise
    7,  # arm roll
    8,  # air drums
    9,  # air guitar
    10,  # other gestures
)

LIGHTINGS = (
    "fluorescent",
    "fluorescent_led",
    "lab",
    "led",
    "natural",
)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dvs_gesture dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Jitter-event conversion of CIFAR10 handwritten digits."),
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(
                        {
                            "time": tfds.features.Tensor(shape=(None,), dtype=np.int64),
                            "coords": tfds.features.Tensor(
                                shape=(None, 2),
                                dtype=np.int64,
                            ),
                            "polarity": tfds.features.Tensor(
                                shape=(None,), dtype=np.bool_
                            ),
                        }
                    ),
                    "label": tfds.features.ClassLabel(names=CLASSES),
                    "user_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                    "lighting": tfds.features.ClassLabel(names=LIGHTINGS),
                }
            ),
            supervised_keys=("events", "label"),
            homepage="https://research.ibm.com/interactive/dvsgesture/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            "https://app.box.com/index.php?rm=box_download_shared_file"
            "&shared_name=3hiq58ww1pbbjrinh367ykfdf60xsfm8&file_id=f_211521748942"
        )
        path = path / "DvsGesture"

        return {
            "train": self._generate_examples(path, "train"),
            "test": self._generate_examples(path, "test"),
        }

    def _generate_examples(self, path: Path, split: str):
        """Yields examples."""
        with open(path / f"trials_to_{split}.txt", "r", encoding="utf-8") as fp:
            filenames = fp.readlines()
        for filename in filenames:
            filename = filename.rstrip()
            if not filename.endswith(".aedat"):
                continue
            example = filename[: -len(".aedat")]
            user, *lighting = example.split("_")
            lighting = "_".join(lighting)
            user = int(user[len("user") :])
            with open(path / f"{example}_labels.csv", "r", encoding="utf-8") as fp:
                label_data = pd.read_csv(fp, header="infer", sep=",")
            labels = np.asarray(label_data["class"]) - 1
            start_times = np.asarray(label_data["startTime_usec"])
            end_times = np.asarray(label_data["endTime_usec"])

            with open(path / filename, "rb", encoding="utf-8") as fp:
                time, x, y, polarity = load_events(fp)
            time = time.astype(np.int64)
            coords = np.stack((x, y), axis=1).astype(np.int64)
            polarity = polarity.astype(bool)

            counts = np.zeros((NUM_CLASSES,), dtype=np.int64)

            for label, start_time, end_time in zip(labels, start_times, end_times):
                mask = np.logical_and(time >= start_time, time <= end_time)
                features = {
                    "events": {
                        "time": time[mask],
                        "coords": coords[mask],
                        "polarity": polarity[mask],
                    },
                    "label": label,
                    "user_id": user,
                    "lighting": lighting,
                }
                key = "_".join([example, CLASSES[label], str(counts[label])])
                counts[label] += 1
                yield key, features
