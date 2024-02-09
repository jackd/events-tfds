# events-tfds

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
import events_tfds.events.nmnist  # pylint:disable=unused-import

dataset = tfds.load("nmnist", split="train", as_supervised=True)
for event, label in dataset:
    coords = event["coords"]
    time = event["time"]
    polarity = event["polarity"]
    do_stuff_with(coords, time, polarity)
```

See [examples](./examples) subdirectory for loading / visualisation scripts.


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
