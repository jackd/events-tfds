import numpy as np

from events_tfds.types import Event


def load_neuro_events(fobj):
    """
    Load events from file.

    File stores concatenated events. Each occupies 40 bits as described below:
        bit 39 - 32: Xaddress (in pixels)
        bit 31 - 24: Yaddress (in pixels)
        bit 23: Polarity (0 for OFF, 1 for ON)
        bit 22 - 0: Timestamp (in microseconds)

    Args:
        fobj: file-like object with a `read` method.

    Returns:
        Event stream, namedtuple with names/shapes:
            time: [num_events] int64
            coords: [num_events, 2] uint8
            polarity: [num_events] bool
    """
    # based on read_dataset from
    # https://github.com/gorchard/event-Python/blob/master/eventvision.py
    raw_data = np.fromstring(fobj.read(), dtype=np.uint8)
    raw_data = raw_data.astype(np.uint32)
    x = raw_data[::5]
    y = raw_data[1::5]
    polarity = ((raw_data[2::5] & 128) >> 7).astype(np.bool)
    time = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    valid = y != 240

    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]
    time = time[valid].astype(np.int64)
    coords = np.stack((x, y), axis=-1).astype(np.int64)

    return Event(time, coords, polarity)
