import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from events_tfds.events.ntidigits import CLASSES

single = False

single_str = "single" if single else "multi"
dataset = tfds.load(f"ntidigits/{single_str}", split="train", as_supervised=True)

for events, labels in dataset:
    time = events["time"].numpy()
    channel = events["channel"].numpy()
    if single:
        print(CLASSES[labels])
    else:
        print([CLASSES[l] for l in labels])
    plt.scatter(channel, time, s=0.1)
    plt.show()
