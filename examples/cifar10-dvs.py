import tensorflow_datasets as tfds

from events_tfds.events.cifar10_dvs import GRID_SHAPE
from events_tfds.vis.anim import animate_frames
from events_tfds.vis.image import as_frames

train_ds = tfds.load("cifar10_dvs", split="train", as_supervised=True)
for events, labels in train_ds:
    coords = events["coords"].numpy()
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()
    coords = coords[:, -1::-1]  # x-y flipped
    frames_kwargs = {
        "coords": coords,
        "time": time,
        "polarity": polarity,
        "shape": GRID_SHAPE,
    }
    frames = as_frames(num_frames=60, **frames_kwargs)
    animate_frames(frames, fps=20, dpi=50)
