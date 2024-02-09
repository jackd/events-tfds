import tensorflow_datasets as tfds

from events_tfds.events.poker_dvs import GRID_SHAPE, CLASSES
from events_tfds.vis.image import as_frames
from events_tfds.vis.anim import animate_frames

dataset = tfds.load("poker_dvs", split="train", as_supervised=True)


for events, labels in dataset:
    coords = events["coords"].numpy()
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()
    frames = as_frames(coords, time, polarity, num_frames=20, shape=GRID_SHAPE)
    print(f"class = {CLASSES[labels.numpy()]}")
    print(f"shape = {tuple(coords.max(axis=0) + 1)}")
    print(f"{time.size} events over {time[-1] - time[0]} dt")
    animate_frames(frames, fps=4)
