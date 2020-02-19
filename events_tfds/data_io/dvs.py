from events_tfds.data_io import aedat


def load_events(fp):
    return aedat.load_events(fp,
                             x_mask=0xfE,
                             x_shift=1,
                             y_mask=0x7f00,
                             y_shift=8,
                             polarity_mask=1,
                             polarity_shift=None)
