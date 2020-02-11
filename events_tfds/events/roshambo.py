from events_tfds.data_io import aedat

if __name__ == '__main__':
    import os
    import numpy as np
    from events_tfds.utils import make_monotonic
    from events_tfds.vis import image
    from events_tfds.vis import anim

    folder = '/home/rslsync/Resilio Sync/RoShamBoNPP/recordings/aedat'
    filename = 'paper_ale_back.aedat'
    path = os.path.join(folder, filename)
    with open(path, 'rb') as fp:
        events = aedat.load_events(fp)

    timestamp, x, y, pol = tuple(d[:1000000] for d in events)

    timestamp = make_monotonic(timestamp, dtype=np.uint64)
    assert (np.all(timestamp[1:] >= timestamp[:-1]))

    print('polarity', np.min(pol), np.max(pol))
    print('x', np.min(x), np.max(x))
    print('y', np.min(y), np.max(y))
    print('num_events: {}'.format(timestamp.size))

    coords = np.stack((x, np.max(y) - y), axis=-1)

    print('Creating animation...')
    frames = image.as_frames(coords, timestamp, pol, num_frames=500)
    anim.animate_frames(frames, fps=10)
