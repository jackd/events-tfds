import numpy as np

RED = np.array(((255, 0, 0)), dtype=np.uint8)
GREEN = np.array(((0, 255, 0)), dtype=np.uint8)
WHITE = np.array(((255, 255, 255)), dtype=np.uint8)


def as_frames(
    coords,
    time,
    polarity=None,
    dt=None,
    num_frames=None,
    shape=None,
    flip_up_down=False,
    clip_head: bool = True,
):
    if time.size == 0:
        raise ValueError("time must not be empty")
    if clip_head:
        time = time - time[0]
    t_end = time[-1]
    assert t_end > 0, time
    if dt is None:
        dt = t_end / (num_frames - 1)
    else:
        num_frames = int(t_end / dt) + 1

    if shape is None:
        shape = np.max(coords, axis=0)[-1::-1] + 1
    else:
        shape = shape[-1::-1]
    frame_data = np.zeros((num_frames, *shape, 3), dtype=np.uint8)
    if polarity is None:
        colors = WHITE
    else:
        colors = np.where(polarity[:, np.newaxis], RED, GREEN)
    i = np.minimum(time / dt, num_frames - 1).astype(np.int64)
    # fi = np.concatenate((i[:, np.newaxis], coords), axis=-1)
    x, y = coords.T
    if flip_up_down:
        y = shape[0] - y - 1
    # frame_data[(i, shape[0] - y - 1, x)] = colors
    assert np.all(i >= 0), (
        i.min(),
        time[-1],
        time[-1] / dt,
        np.array(time[-1] / dt, np.int64),
    )
    assert np.all(y >= 0), y.min()
    assert np.all(x >= 0), x.min()
    frame_data[(i, y, x)] = colors

    return frame_data


def as_frame(coords, polarity, shape=None, out=None):
    if shape is None:
        if out is None:
            shape = np.max(coords, axis=0)[-1::-1] + 1
        else:
            shape = out.shape

    x, y = coords.T
    if out is None:
        out = np.zeros((*shape, 3), dtype=np.uint8)
    out[(shape[0] - y - 1, x)] = np.where(polarity[:, np.newaxis], RED, GREEN)
    return out
