import tensorflow_datasets as tfds

from events_tfds.events.nmnist import GRID_SHAPE
from events_tfds.vis.image import as_frames
from events_tfds.vis.anim import animate_frames

dataset = tfds.load("nmnist", split="train", as_supervised=True)


for events, labels in dataset:
    frames = as_frames(
        **{k: v.numpy() for k, v in events.items()}, num_frames=20, shape=GRID_SHAPE
    )
    print(labels.numpy())
    t = events["time"].numpy()
    print(t[-1] - t[0], len(t))
    animate_frames(frames, fps=4)
