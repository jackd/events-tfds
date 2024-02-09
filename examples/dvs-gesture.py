import tensorflow_datasets as tfds

from events_tfds.events.dvs_gesture import GRID_SHAPE, CLASSES
from events_tfds.vis.anim import animate_frames
from events_tfds.vis.image import as_frames

train_ds = tfds.load("dvs_gesture", split="train", as_supervised=True)
for events, labels in train_ds:
    coords = events["coords"].numpy()
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()
    frames_kwargs = {
        "coords": coords,
        "time": time,
        "polarity": polarity,
        "shape": GRID_SHAPE,
    }
    print(f"class = {CLASSES[labels.numpy()]}")
    print(f"shape = {tuple(coords.max(axis=0) + 1)}")
    print(f"{time.size} events over {time[-1] - time[0]} dt")
    frames = as_frames(num_frames=60, **frames_kwargs)
    animate_frames(frames, fps=20, dpi=50)
