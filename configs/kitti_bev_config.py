color_map = (0, 0, 142)
OBJ_CLASSES = ['Car']
data_config = {
    'data_root': '/data/kitti',
    'data_type': 'train',
    'bev_config':{
        'height_lo': -0.2,
        'height_hi': 2.3,
        'num_slices': 5,
        'voxel_size': 0.1,
        'area_extents':[[-40, 40], [-5, 3], [0, 70]], # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        'density_threshold': 1,
    },
    'camera_baseline': 0.54
}

MODEL_CONFIG = {
    'num_classes': 2,
    'num_anchors': 6,

    'output_stride': [4., 8.],
    # 'default_ratio': [0.0449, 0.0772, 0.115, 0.164, 0.227],
    'default_ratio':[0.034, 0.034, 0.034],
    'aspect_ratio': ((2.39, ), (2.39, )),
    'input_shape': (800, 700),
}
