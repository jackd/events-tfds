import os

import numpy as np

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def load_raw_events(
    fp, bytes_skip=0, bytes_trim=0, filter_dvs=False, times_first=False
):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.fromstring(data, dtype=">u4")
    if len(data) % 2 != 0:
        print(data[:20:2])
        print("---")
        print(data[1:21:2])
        raise ValueError("odd number of data elements")
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    return timestamp, raw_addr


def parse_raw_address(
    addr,
    x_mask=x_mask,
    x_shift=x_shift,
    y_mask=y_mask,
    y_shift=y_shift,
    polarity_mask=polarity_mask,
    polarity_shift=polarity_shift,
):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def load_events(
    fp,
    filter_dvs=False,
    # bytes_skip=0,
    # bytes_trim=0,
    # times_first=False,
    **kwargs
):
    timestamp, addr = load_raw_events(
        fp,
        filter_dvs=filter_dvs,
        #   bytes_skip=bytes_skip,
        #   bytes_trim=bytes_trim,
        #   times_first=times_first
    )
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity


if __name__ == "__main__":
    from events_tfds.utils import make_monotonic

    folder = "/home/rslsync/Resilio Sync/RoShamBoNPP/recordings/aedat/"
    # filename = 'background_10.aedat'
    filename = "paper_tobi_front.aedat"
    path = os.path.join(folder, filename)
    with open(path, "rb") as fp:
        time, x, y, pol = load_events(fp)
    time = make_monotonic(time)
    time -= np.min(time)
    x -= np.min(x)
    y -= np.min(y)
    from events_tfds.vis import image
    from events_tfds.vis import anim

    coords = np.stack((x, y), axis=-1)
    frames = image.as_frames(coords, time, pol, num_frames=100)
    anim.animate_frames(frames, 20)
