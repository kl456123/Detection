# -*- coding: utf-8 -*-

from utils.box_vis import draw_line
import numpy as np

image_path = '/data/object/training/image_2/000052.png'
a = np.asarray([[[740.2719, 243.6114],
             [711.2286, 230.7807]], [[625.7795, 239.5613],
                                     [631.7291, 227.8167]],
            [[321.1204, 217.5381],
             [406.7209, 217.0684]], [[69.1882, 209.3627], [74.0953, 212.0611]],
            [[310.9117, 248.0898], [378.7721, 260.0074]]])
draw_line(image_path, a)
