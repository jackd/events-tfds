import tensorflow_datasets as tfds

from events_tfds.events.ncaltech101 import load_class_names
from events_tfds.vis.image import as_frames
from events_tfds.vis.anim import animate_frames

class_names = load_class_names()

dataset = tfds.load("ncaltech101", split="train", as_supervised=True)

for events, labels in dataset:
    coords = events["coords"].numpy()
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()
    frames = as_frames(coords, time, polarity, num_frames=20)
    print(f"class = {class_names[labels.numpy()]}")
    print(f"shape = {tuple(coords.max(axis=0) + 1)}")
    print(f"{time.size} events over {time[-1] - time[0]} dt")
    animate_frames(frames, fps=4)
