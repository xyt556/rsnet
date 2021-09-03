import numpy as np
from PIL import Image
# viz
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt

from rsnet.converter import polygonize


def set_axis(ax, x0, xN, y0, yN, title):
    ax.set_xlim(x0, xN)
    ax.set_ylim(y0, yN)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title)


cls_map = np.array(Image.open('data/cls_map2.png'))
height, width = cls_map.shape[:2]

fig = plt.figure(1, figsize=(10, 8))
ax = fig.add_subplot(221)
ax.imshow(cls_map)
set_axis(ax, 0.2 * width, 0.9 * width, 0.2 * height, 0.9 * width, 'cls map')

#
BLUE = '#6699cc'
GRAY = '#999999'
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
shapes = polygonize(cls_map, mask=cls_map == 1, simplify=False)
for s, v in shapes:
    patch = PolygonPatch(s, fc='gray', ec='blue', alpha=0.5, zorder=2)
    ax1.add_patch(patch)
    ax3.add_patch(PolygonPatch(s, fc='#ffffff', ec='blue', zorder=1))

shapes = polygonize(cls_map, mask=cls_map == 1, simplify=True, epsilon=20)
for s, v in shapes:
    patch = PolygonPatch(s, fc='gray', ec='blue', alpha=0.5, zorder=2)
    ax2.add_patch(patch)
    ax3.add_patch(PolygonPatch(s, fc='#ffffff', ec='red', zorder=1))

set_axis(ax1, 0.2 * width, 0.9 * width, 0.2 * height, 0.9 * width,
         'polygonize')
set_axis(ax2, 0.2 * width, 0.9 * width, 0.2 * height, 0.9 * width,
         'simplified')
set_axis(ax3, 0.2 * width, 0.9 * width, 0.2 * height, 0.9 * width, 'compare')

plt.show()
