import os.path as osp

import rasterio as rio
from affine import Affine
from tqdm import tqdm
import numpy as np

from ..dataset import RasterSampleDataset
from ..utils import mkdir


def window_transform(window, transform):
    """Construct an affine transform matrix relative to a window.
    
    Args:
        window (Window): The input window.
        transform (Affine): an affine transform matrix.
    Returns:
        (Affine): The affine transform matrix for the given window
    """
    x, y = transform * (window.col_off, window.row_off)
    return Affine.translation(x - transform.c, y - transform.f) * transform


class RasterDataSpliter(RasterSampleDataset):
    def __init__(self,
                 fname,
                 win_size,
                 step_size,
                 suffix_tmpl='_{}_{}',
                 to_type='uint8'):
        super().__init__(fname=fname,
                         win_size=win_size,
                         step_size=step_size,
                         pad_size=0,
                         to_type=to_type)

        self.suffix_tmpl = suffix_tmpl

    def run(self, outpath, progress=True):
        mkdir(outpath)
        basename = self.name
        suffix = f'{self.suffix_tmpl}.{self.suffix}'
        meta = self.meta
        width, height = self.win_size

        pbar = self.window_ids
        if progress:
            pbar = tqdm(pbar)
        for x, y in pbar:
            tile, window = self.sample(x, y)
            transform = window_transform(window, self.affine_matrix)

            xoff, yoff = window.col_off, window.row_off
            outfile = osp.join(outpath, basename + suffix.format(xoff, yoff))
            meta.update(width=width,
                        height=height,
                        transform=transform,
                        dtype=np.dtype(self.to_type))
            with rio.open(outfile, 'w', **meta) as dst:
                dst.write(tile.transpose(2, 0, 1))

    def sample(self, x, y):
        xmin, ymin = x, y
        xsize, ysize = self.win_size
        window = rio.windows.Window(xmin, ymin, xsize, ysize)
        tile = super().sample(x, y)

        return tile, window
