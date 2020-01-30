import collections
import numpy as np

Event = collections.namedtuple('Event', ('time', 'coords', 'polarity'))


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
    time = (
        (raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    valid = y != 240

    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]
    time = time[valid].astype(np.int64)
    coords = np.stack((x, y), axis=-1).astype(np.int64)

    return Event(time, coords, polarity)


def as_frames(coords, time, polarity, dt=None, num_frames=None):
    t_start = time[0]
    t_end = time[-1]
    if num_frames is None:
        assert (dt is not None)
        frame_times = np.arange(t_start, t_end, dt)
    else:
        frame_times = np.linspace(t_start, t_end, num_frames)
    starts = frame_times[:-1]
    ends = frame_times[1:]

    num_frames = starts.size
    shape = np.max(coords, axis=0)[-1::-1] + 1
    frame_data = np.zeros((num_frames, *shape, 3), dtype=np.uint8)

    RED = np.array(((255, 0, 0)), dtype=np.uint8)
    GREEN = np.array(((0, 255, 0)), dtype=np.uint8)
    for i, (start, end) in enumerate(zip(starts, ends)):
        mask = np.logical_and(time >= start, time < end)
        x, y = coords[mask].T
        pol = polarity[mask]
        colors = np.where(pol[:, np.newaxis], RED, GREEN)
        frame_data[i][y, x] = colors
    return frame_data


def animate_frames(img_data, fps=30, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    im = plt.imshow(img_data[0])

    def init():
        im.set_data(img_data[0])
        return im,

    def animate(i):
        im.set_data(img_data[i])
        return im,

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=img_data.shape[0],
        interval=1000 // fps,  # in ms
    )
    if save_path is None:
        plt.show()
    else:
        anim.save(save_path)
