# -*- coding: utf-8 -*-
# label
KEY_IMAGE = 'image'
KEY_LABEL_BOXES_2D = 'label_boxes_2d'
KEY_LABEL_BOXES_3D = 'label_boxes_3d'
KEY_LABEL_CLASSES = 'label_classes'
KEY_LABEL_ORIENTS = 'label_orients'
KEY_IMAGE_PATH = 'image_path'
KEY_IMAGE_INFO = 'image_info'
KEY_NUM_INSTANCES = 'num_instances'
KEY_OBJECTNESS = 'objectness'

KEY_STEREO_CALIB_P2 = 'stereo_calib_p2'
KEY_STEREO_CALIB_P2_ORIG = 'stereo_calib_p2_orig'

# logger config key
KEY_LOGGER_NAME = 'logger_name'
KEY_LOGGER_LEVEL = 'logger_level'
KEY_LOGGER_PATH = 'logger_path'

# pred
KEY_TARGETS = 'targets'

# all attributions of instance
KEY_PRIMARY = 'primary'
# KEY_NON_PRIME = 'non_prime'
# KEY_PRED_CLASSES = 'pred_classes'
# KEY_PRED_BOXES_2D = 'pred_boxes_2d'
# KEY_PRED_BOXES_3D = 'pred_boxes_3d'

# attribution fields of instance
KEY_BOXES_2D = 'boxes_2d'
KEY_BOXES_3D = 'boxes_3d'
KEY_CLASSES = 'classes'
KEY_KEYPOINTS = 'keypoints'
KEY_KEYPOINTS_HEATMAP = 'keypoints_heatmap'
KEY_KEYPOINTS_DENSE_REG = 'keypoints_dense_reg'
KEY_ORIENTS = 'orients'
KEY_ORIENTS_V2 = 'orients_v2'
KEY_ORIENTS_V3 = 'orients_v3'
KEY_REAR_SIDE = 'rear_side'
KEY_BOXES_2D_REFINE = 'boxes_2d_refine'
KEY_DIMS = 'dims'
KEY_FINAL_PROBS = 'final_probs'
KEY_CORNERS_2D = 'corners_2d'
KEY_CORNERS_3D = 'corners_3d'
KEY_CORNERS_3D_GRNET = 'corners_3d_grnet'
KEY_CORNERS_3D_BETTER = 'corners_3d_better'
KEY_CORNERS_2D_NEAREST = 'corners_2d_nearest'
KEY_CORNERS_2D_NEAREST_DEPTH = 'corners_2d_nearest_depth'
KEY_CORNERS_VISIBILITY = 'corners_visibility'
KEY_CORNERS_2D_HM = 'corners_2d_hm'
KEY_CORNERS_2D_NEARESTV2 = 'corners_2d_nearest_v2'
KEY_MONO_3D_NON_2D_PROJ = 'mono_3d_non_2d_proj'
KEY_MONO_3D_2D_PROJ = 'mono_3d_2d_proj'
KEY_CORNERS_2D_STABLE = 'corners_2d_stable'

# key of stats
KEY_STATS_RECALL = 'stats_recall'
KEY_STATS_PRECISION = 'stats_precision'
KEY_STATS_THRESH_RECALL = 'stats_thresh_recall'
KEY_STATS_ORIENTS_PRECISION = 'stats_orients_precision'
KEY_STATS = 'stats'

# auxiliary dict
KEY_PROPOSALS = 'proposals'
KEY_MATCH = 'match'
KEY_IGNORED_MATCH = 'ignored_match'
KEY_BG_THRESH = 'bg_thresh'
KEY_ASSIGNED_OVERLAPS = 'assigned_overlaps'
KEY_MEAN_DIMS = 'mean_dims'
KEY_FAKE_MATCH = 'fake_match'
KEY_TARGET_GENERATOR_CONFIG = 'target_generator_config'
