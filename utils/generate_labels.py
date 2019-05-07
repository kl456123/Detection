# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import json
from nuscenes.nuscenes import NuScenes
from box_vis import draw_boxes
from PIL import Image, ImageDraw
"""
label format
file_format: json
content_format:
{
"image_name":{
"name":"image_name",
"category":[],
"box_3d":[[ry, hwl, xyz]]
},
...}
"""

classes_map = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    'human.pedestrian.adult': "pedestrian",
    'human.pedestrian.child': "pedestrian",
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}
# SOME CONFIGS
DATA_ROOT = '/home/breakpoint/data/nuScenes/'
SENSOR = 'CAM_FRONT'

nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT, verbose=True)


def rotate(theta):
    if theta < 0:
        return -np.pi - theta
    return np.pi - theta


def get_ry(rotation):
    # w, z, x, y = rotation
    w, x, y, z = rotation
    # w = rotation[0]
    # x = rotation[3]
    # y = rotation[1]
    # z = rotation[2]
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    theta = np.arcsin(2 * (w * y - x * z))
    alpha = np.arctan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    eluer_angle = [phi, theta, alpha]

    if np.fabs(alpha) > 0.5 * np.pi:
        return rotate(theta)
    return theta
    # return eluer_angle[1]
    # return 0

# theta = 2 * np.arccos(w)
# if z < 0:

# return -theta
# return theta


calib_format = [
    "P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00",
    "P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00",
    "P2: 1253.17216411 0.00000000e+00 609.99510914999996 0.00000000e+00 0.00000000e+00 1252.4289983399999 363.50622271999998 0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00",
    "P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03",
    "R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01",
    "Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01",
    "Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01"
]


def save_calib(camera_intrinsic, fn):
    P2 = camera_intrinsic.reshape(-1).tolist()
    P2 = [str(item) for item in P2]
    P2_str = "P2: " + " ".join(P2)
    calib_format[2] = P2_str
    with open(fn, 'w') as f:
        f.write('\n'.join(calib_format))


def get_trans(trans, h):
    # return [-trans[1], -trans[2], trans[0]]
    return [trans[0], trans[1] + 0.5 * h, trans[2]]
    # return trans


def get_dim(dim):
    # old (wlh) (yxz) (left, forward, up)
    # kitti (hwl) (up, forward ,left)(dim[::-1])
    # kitti (wlh) (forward, left, up)([1, 0, 2])
    # return dim[::-1]
    # return dim[[1, 0, 2]]
    return dim[[2, 0, 1]]
    # return dim


def visualize(index):
    sample = nusc.sample[index]
    my_sample = nusc.get('sample', sample['token'])
    nusc.render_sample_data(
        my_sample['data'][SENSOR], out_path='images/{:06d}'.format(index))


# for ind, sample in enumerate(nusc.sample):
# visualize(ind)
# sys.stdout.write('\r{}'.format(ind))
# sys.stdout.flush()

res_json = dict()

num_samples = len(nusc.sample)

for ind, sample in enumerate(nusc.sample):
    sample_json = dict()
    category = []
    box_3d = []
    rotations = []
    my_sample = nusc.get('sample', sample['token'])
    sample_data = nusc.get('sample_data', my_sample['data'][SENSOR])
    filename = os.path.basename(sample_data['filename'])

    my_annotation_token_list = my_sample['anns']
    sensor_modality = sample_data['sensor_modality']
    assert sensor_modality == 'camera'
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(
        my_sample['data'][SENSOR])

    # for anno_token in my_annotation_token_list:
    # my_annotation_metadata = nusc.get('sample_annotation', anno_token)
    # # class
    # cat_name = my_annotation_metadata['category_name']
    for box in boxes:
        cat_name = box.name
        if classes_map.get(cat_name) is None:
            continue
        else:
            category.append(classes_map[cat_name])
        # xyz (forward, left, up)
        trans = box.center
        # wlh (yxz) (left, forward, up)
        dim = box.wlh
        rotation = box.orientation
        ry = get_ry(rotation)
        rotations.append(box.rotation_matrix)

        # convert to kitti format
        trans_kitti = get_trans(trans, dim[2])
        dim_kitti = get_dim(dim)
        anno_kitti = []
        anno_kitti.append(ry)
        anno_kitti.extend(dim_kitti)
        anno_kitti.extend(trans_kitti)
        box_3d.append(anno_kitti)
    # import ipdb
    # ipdb.set_trace()
    # visualize(ind)
    p2 = np.concatenate(
        [camera_intrinsic, np.array([0, 0, 0]).reshape(3, 1)], axis=-1)
    # rotation_matrix = np.asarray(
    # [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    # dtype=np.float32).reshape(4, 4)
    # p2 = np.dot(p2, rotation_matrix)
    img = Image.open(data_path)
    # import ipdb
    # ipdb.set_trace()
    draw_boxes(
        img,
        np.asarray(box_3d),
        p2,
        save_path=os.path.basename(data_path),
        box_3d_gt=None)

    # import ipdb
    # ipdb.set_trace()
    sample_json['name'] = os.path.basename(data_path)
    sample_json['box_3d'] = box_3d
    sample_json['category'] = category
    res_json[filename] = sample_json
    fn = os.path.join('calibs',
                      '{}.txt'.format(os.path.basename(data_path)[:-4]))
    save_calib(p2, fn)
    sys.stdout.write('\r{}/{}'.format(ind, num_samples))
    sys.stdout.flush()

with open('trainval.json', 'w') as f:
    json.dump(res_json, f)

print('Done !')
