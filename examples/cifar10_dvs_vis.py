import tensorflow_datasets as tfds

import events_tfds.events.cifar10_dvs  # pylint:disable=unused-import
from events_tfds.vis.anim import animate_frames
from events_tfds.vis.image import as_frames

train_ds = tfds.load("cifar10_dvs", split="train", as_supervised=True)
for events, labels in train_ds:
    coords = events["coords"].numpy()
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()
    coords = coords[:, -1::-1]  # x-y flipped
    frames = as_frames(coords=coords, time=time, polarity=polarity, num_frames=20)
    animate_frames(frames, fps=4)
