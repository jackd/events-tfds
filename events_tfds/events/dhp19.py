import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from events_tfds.data_io import aedat

JOINTS = (
    "head",
    "shoulderR",
    "shoulderL",
    "elbowR",
    "elbowL",
    "hipL",
    "hipR",
    "handR",
    "handL",
    "kneeR",
    "kneeL",
    "footR",
    "footL",
)

NUM_JOINTS = len(JOINTS)
assert NUM_JOINTS == 13


def load_vicon_data(filename):
    with tf.io.gfile.GFile(filename, "rb") as fp:
        data = loadmat(fp)
    return np.array(tuple(data["XYZPOS"][0, 0]))


camera_mask = 0xF  # not just be 0x4 for 2 least significant bits??
camera_shift = None


def load_dhp19_events(fp):
    timestamp, addr = aedat.load_raw_events(fp, filter_dvs=True)
    x, y, polarity = aedat.parse_raw_address(addr)
    camera = aedat.read_bits(addr, camera_mask, camera_shift)
    return timestamp, x, y, polarity, camera


if __name__ == "__main__":
    import os
    from events_tfds.utils import make_monotonic
    from events_tfds.vis import image
    from events_tfds.vis import anim

    folder = "/home/rslsync/Resilio Sync/DHP19/DVS_movies/S1/session1"
    filename = "mov8.aedat"
    path = os.path.join(folder, filename)
    with tf.io.gfile.GFile(path, "rb") as fp:
        timestamp, x, y, pol, camera = load_dhp19_events(fp)

    timestamp = make_monotonic(timestamp, dtype=np.uint64)
    assert np.all(timestamp[1:] >= timestamp[:-1])

    print("camera", np.min(camera), np.max(camera))
    print("polarity", np.min(pol), np.max(pol))
    print("x", np.min(x), np.max(x))
    print("y", np.min(y), np.max(y))
    cam_mask = camera == 1
    pol = pol[cam_mask]
    x = x[cam_mask]
    y = y[cam_mask]
    timestamp = timestamp[cam_mask]
    print(f"num_events: {timestamp.size}")

    coords = np.stack((x, np.max(y) - y), axis=-1)

    print("Creating animation...")
    frames = image.as_frames(coords, timestamp, pol, num_frames=200)
    anim.animate_frames(frames, fps=20)
