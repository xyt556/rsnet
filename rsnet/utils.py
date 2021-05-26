from typing import Iterable
from itertools import repeat
import os
import os.path as osp

import numpy as np


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


pair = _ntuple(2)


def mkdir(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def bytescale(band, mask, cmin, cmax, dtype=np.uint8):
    band = np.asfarray(band)
    dtype_in = band.dtype
    dtype_out = np.dtype(dtype)
    mask = np.asarray(mask).astype(np.bool)

    if dtype_in == dtype_out:
        return band
    imin_out = np.iinfo(dtype_out).min
    imax_out = np.iinfo(dtype_out).max

    cmin = np.asarray(cmin)
    cmax = np.asarray(cmax)

    imin_out = cmin / cmax * imax_out
    band = (band - cmin) / (cmax - cmin) * (imax_out - imin_out) + imin_out
    # print(band.max(axis=(1, 2)))
    band = (band.clip(imin_out, imax_out) * mask).astype(dtype_out)

    return band
