# events-fds

[tensorflow-datasets](https://github.com/tensorflow/datasets) implementations of event stream datasets, along with basic implementations.

## Installation

```bash
# install tensorflow somehow
git clone https://github.com/jackd/events-tfds
pip install events-tfds
```

## Example Usage

```python
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


```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
