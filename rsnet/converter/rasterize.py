import fiona
import rasterio
from rasterio import features


def rasterize(file, like, property='id', **kwargs):
    """Rasterize vector.
    """
    with fiona.open(file) as src:
        shapes = ((f['geometry'], int(f['properties'][property])) for f in src)

        with rasterio.open(like) as ref:
            out_shape = ref.shape
            transform = ref.transform

        image = features.rasterize(shapes,
                                   out_shape=out_shape,
                                   fill=0,
                                   transform=transform)

        return image
