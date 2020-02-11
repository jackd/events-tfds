import numpy as np


def make_monotonic(wrapped, dtype=np.uint64):
    unwrapped = np.empty(wrapped.shape, dtype=dtype)
    unwrapped[1:] = np.cumsum((wrapped[1:] < wrapped[:-1]).astype(dtype))
    unwrapped[0] = 0
    unwrapped *= np.iinfo(wrapped.dtype).max
    unwrapped += wrapped
    return unwrapped
