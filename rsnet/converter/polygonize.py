from rasterio import features
from affine import Affine
from shapely import geometry

from simplification.cutil import simplify_coords_vwp, simplify_coords

IDENTITY = Affine.identity()
GDAL_IDENTITY = IDENTITY.to_gdal()


def polygonize(cls_map,
               mask,
               connectivity=4,
               transform=IDENTITY,
               simplify=False,
               epsilon=20):
    """

    Args:
        cls_map:
        mask:
        connectivity:
        transform:
    
    Returns:
        shapes (Generator): A pair of (polygon, value) for each feature 
        found in the image. Polygons are GeoJSON-like dicts and the values 
        are the associated value from the image, in the data type of the image.
    """
    shapes = features.shapes(cls_map,
                             mask=mask,
                             connectivity=connectivity,
                             transform=transform)

    for s, v in shapes:
        if simplify:
            s = geometry.shape(s)
            s = polygon_simplication(s, epsilon=epsilon)
            if s is None:
                continue
        yield s, v


def polygon_simplication(polygon, epsilon=30, method='vw'):
    """
    
    Args:
        method (str): 'vm' or 'dp'
    """
    exterior_coords = polygon.exterior.coords
    interior_coords = []
    for interior in polygon.interiors:
        interior_coords.append(interior.coords[:])

    # simplify
    exterior_coords = simplify_coords_vwp(exterior_coords, epsilon)
    new_interior_coords = []
    for interior in interior_coords:
        simplify_coords = simplify_coords_vwp(interior, epsilon)
        if len(simplify_coords) < 4:
            continue
        new_interior_coords.append(simplify_coords)

    if len(exterior_coords) < 4:
        return None
    # construct new polygon
    poly = geometry.Polygon(exterior_coords, new_interior_coords)

    return poly
