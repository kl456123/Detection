# -*- coding: utf-8 -*-

from utils import geometry_utils
import numpy as np
boxes_3d = np.asarray([[2.3900,   1.6000,   5.3700, 3.7800, 1.4200, 1.6000, 0.5 * np.pi]])
p2 = np.asarray([[721.5377, 0.0000, 609.5593, 44.8573],
                 [0.0000, 721.5377, 172.8540, 0.2164],
                 [0.0000, 0.0000, 1.0000, 0.0027]])

corners_2d = geometry_utils.boxes_3d_to_corners_2d(boxes_3d, p2)
boxes_2d = geometry_utils.corners_2d_to_boxes_2d(corners_2d)
print(boxes_2d)
