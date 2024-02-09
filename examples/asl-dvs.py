import tensorflow_datasets as tfds

from events_tfds.events.asl_dvs import CLASSES, GRID_SHAPE
from events_tfds.vis.image import as_frames
from events_tfds.vis.anim import animate_frames

# save_path = "/tmp/asl-dvs.gif"
save_path = None
dataset = tfds.load("asl_dvs", split="train", as_supervised=True)


kwargs = {"flip_up_down": True, "shape": GRID_SHAPE}

for events, labels in dataset:
    coords = events["coords"].numpy()
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()
    print(coords.max(0) + 1)
    print(f"{time.size} events over dt = {time[-1] - time[0]}")
    print(f"class = {CLASSES[labels.numpy()]}")
    frames = as_frames(coords, time, polarity, num_frames=24, **kwargs)
    animate_frames(frames, fps=8, save_path=save_path)
    if save_path is not None:
        print(f"Animation saved to {save_path}")
        break
