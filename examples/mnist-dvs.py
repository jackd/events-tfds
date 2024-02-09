import tensorflow_datasets as tfds
from events_tfds.events.mnist_dvs import GRID_SHAPE

from events_tfds.vis.image import as_frames
from events_tfds.vis.anim import animate_frames

dataset = tfds.load("mnist_dvs/scale16", split="train", as_supervised=True)

for events, labels in dataset:
    coords = events["coords"].numpy()
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()
    frames = as_frames(coords, time, polarity, num_frames=40, shape=GRID_SHAPE)
    print(f"label = {labels.numpy()}")
    print(f"shape = {tuple(coords.max(0) + 1)}")
    time = events["time"].numpy()
    print(f"{time.shape[0]} events over {time[-1] - time[0]} dt")
    animate_frames(frames, fps=8)
