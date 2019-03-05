import numpy as np
import torch

from wavedata.tools.core import geometry_utils

from avod.core import box_3d_encoder

"""Box4c Encoder
Converts boxes between the box_3d and box_4c formats.
- box_4c format: [x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]
- corners are in the xz plane, numbered clockwise starting at the top right
- h1 is the height above the ground plane to the bottom of the box
- h2 is the height above the ground plane to the top of the box
"""


def np_box_3d_to_box_4c(box_3d, ground_plane):
    """Converts a single box_3d to box_4c

    Args:
        box_3d: box_3d (6,)
        ground_plane: ground plane coefficients (4,)

    Returns:
        box_4c (10,)
    """
    anchor = box_3d_encoder.box_3d_to_anchor(box_3d, ortho_rotate=True)[0]

    centroid_x = anchor[0]
    centroid_y = anchor[1]
    centroid_z = anchor[2]
    dim_x = anchor[3]
    dim_y = anchor[4]
    dim_z = anchor[5]

    # Create temporary box at (0, 0) for rotation
    half_dim_x = dim_x / 2
    half_dim_z = dim_z / 2

    # Box corners
    x_corners = np.asarray([half_dim_x, half_dim_x,
                            -half_dim_x, -half_dim_x])

    z_corners = np.array([half_dim_z, -half_dim_z,
                          -half_dim_z, half_dim_z])

    ry = box_3d[6]

    # Find nearest 90 degree
    half_pi = np.pi / 2
    ortho_ry = np.round(ry / half_pi) * half_pi

    # Find rotation to make the box ortho aligned
    ry_diff = ry - ortho_ry

    # Create transformation matrix, including rotation and translation
    tr_mat = np.array([[np.cos(ry_diff), np.sin(ry_diff), centroid_x],
                       [-np.sin(ry_diff), np.cos(ry_diff), centroid_z],
                       [0, 0, 1]])

    # Create a ones row
    ones_row = np.ones(x_corners.shape)

    # Append the column of ones to be able to multiply
    points_stacked = np.vstack([x_corners, z_corners, ones_row])
    corners = np.matmul(tr_mat, points_stacked)

    # Discard the last row (ones)
    corners = corners[0:2]

    # Calculate height off ground plane
    ground_y = geometry_utils.calculate_plane_point(
        ground_plane, [centroid_x, None, centroid_z])[1]
    h1 = ground_y - centroid_y
    h2 = h1 + dim_y

    # Stack into (10,) ndarray
    box_4c = np.hstack([corners.flatten(), h1, h2])
    return box_4c

def np_box_4c_to_box_3d(box_4c, ground_plane):
    """Converts a single box_4c to box_3d. The longest midpoint-midpoint
    length is used to calculate orientation. Points are projected onto the
    orientation vector and the orthogonal vector to get the bounding box_3d.
    The centroid is calculated by adding a vector of half the projected length
    along the midpoint-midpoint vector, and a vector of the width
    differences along the normal.

    Args:
        box_4c: box_4c to convert (10,)
        ground_plane: ground plane coefficients (4,)

    Returns:
        box_3d (7,)
    """

    # Extract corners
    corners = box_4c[0:8].reshape(2, 4)

    p1 = corners[:, 0]
    p2 = corners[:, 1]
    p3 = corners[:, 2]
    p4 = corners[:, 3]

    # Check for longest axis
    midpoint_12 = (p1 + p2) / 2.0
    midpoint_23 = (p2 + p3) / 2.0
    midpoint_34 = (p3 + p4) / 2.0
    midpoint_14 = (p1 + p4) / 2.0

    vec_34_12 = midpoint_12 - midpoint_34
    vec_34_12_mag = np.linalg.norm(vec_34_12)

    vec_23_14 = midpoint_14 - midpoint_23
    vec_23_14_mag = np.linalg.norm(vec_23_14)

    # Check which midpoint -> midpoint vector is longer
    if vec_34_12_mag > vec_23_14_mag:
        # vec_34_12_mag longer
        vec_34_12_norm = vec_34_12 / vec_34_12_mag

        vec_mid_34_p1 = p1 - midpoint_34
        vec_mid_34_p2 = p2 - midpoint_34
        vec_mid_34_p3 = p3 - midpoint_34
        vec_mid_34_p4 = p4 - midpoint_34

        l1 = np.dot(vec_mid_34_p1, vec_34_12_norm)
        l2 = np.dot(vec_mid_34_p2, vec_34_12_norm)
        l3 = np.dot(vec_mid_34_p3, vec_34_12_norm)
        l4 = np.dot(vec_mid_34_p4, vec_34_12_norm)
        all_lengths = [l1, l2, l3, l4]

        min_l = np.amin(all_lengths)
        max_l = np.amax(all_lengths)
        length_out = max_l - min_l

        ortho_norm = np.asarray([-vec_34_12_norm[1], vec_34_12_norm[0]])
        w1 = np.dot(vec_mid_34_p1, ortho_norm)
        w2 = np.dot(vec_mid_34_p2, ortho_norm)
        w3 = np.dot(vec_mid_34_p3, ortho_norm)
        w4 = np.dot(vec_mid_34_p4, ortho_norm)
        all_widths = [w1, w2, w3, w4]

        min_w = np.amin(all_widths)
        max_w = np.amax(all_widths)
        w_diff = max_w + min_w
        width_out = max_w - min_w

        ry_out = -np.arctan2(vec_34_12[1], vec_34_12[0])

        # New centroid
        centroid = midpoint_34 + vec_34_12_norm * (min_l + max_l) / 2.0 + \
                   ortho_norm * w_diff

    else:
        # vec_23_14_mag longer
        vec_23_14_norm = vec_23_14 / vec_23_14_mag

        vec_mid_23_p1 = p1 - midpoint_23
        vec_mid_23_p2 = p2 - midpoint_23
        vec_mid_23_p3 = p3 - midpoint_23
        vec_mid_23_p4 = p4 - midpoint_23

        l1 = np.dot(vec_mid_23_p1, vec_23_14_norm)
        l2 = np.dot(vec_mid_23_p2, vec_23_14_norm)
        l3 = np.dot(vec_mid_23_p3, vec_23_14_norm)
        l4 = np.dot(vec_mid_23_p4, vec_23_14_norm)
        all_lengths = [l1, l2, l3, l4]

        min_l = np.amin(all_lengths)
        max_l = np.amax(all_lengths)

        length_out = max_l - min_l

        ortho_norm = np.asarray([-vec_23_14_norm[1], vec_23_14_norm[0]])
        w1 = np.dot(vec_mid_23_p1, ortho_norm)
        w2 = np.dot(vec_mid_23_p2, ortho_norm)
        w3 = np.dot(vec_mid_23_p3, ortho_norm)
        w4 = np.dot(vec_mid_23_p4, ortho_norm)
        all_widths = [w1, w2, w3, w4]

        min_w = np.amin(all_widths)
        max_w = np.amax(all_widths)
        w_diff = max_w + min_w
        width_out = max_w - min_w

        ry_out = -np.arctan2(vec_23_14[1], vec_23_14[0])

        # New centroid
        centroid = midpoint_23 + vec_23_14_norm * (min_l + max_l) / 2.0 + \
                   ortho_norm * w_diff

    # Find new centroid y
    a = ground_plane[0]
    b = ground_plane[1]
    c = ground_plane[2]
    d = ground_plane[3]

    h1 = box_4c[8]
    h2 = box_4c[9]

    centroid_x = centroid[0]
    centroid_z = centroid[1]

    ground_y = -(a * centroid_x + c * centroid_z + d) / b

    # h1 and h2 are along the -y axis
    centroid_y = ground_y - h1
    height_out = h2 - h1

    box_3d_out = np.stack([centroid_x, centroid_y, centroid_z,
                           length_out, width_out, height_out, ry_out])

    return box_3d_out

def tensor_box_3d_to_box_4c(boxes_3d, ground_plane):
    anchors = torch.FloatTensor(box_3d_encoder.box_3d_to_anchor(boxes_3d))

    centroid_x = anchors[:, 0]
    centroid_y = anchors[:, 1]
    centroid_z = anchors[:, 2]
    dim_x = anchors[:, 3]
    dim_y = anchors[:, 4]
    dim_z = anchors[:, 5]

    # Create temporary box at (0, 0) for rotation
    half_dim_x = dim_x / 2
    half_dim_z = dim_z / 2

    x_corners = torch.stack((half_dim_x, half_dim_x,
                            -half_dim_x, -half_dim_x), dim=1)
    z_corners = torch.stack((half_dim_z, -half_dim_z,
                          -half_dim_z, half_dim_z), dim=1)

    all_rys = boxes_3d[:, 6]
    half_pi = np.pi / 2
    ortho_rys = np.round(all_rys / half_pi) * half_pi

    ry_diffs = all_rys - ortho_rys
    zeros = torch.zeros_like(ry_diffs).float()
    ones = torch.ones_like(ry_diffs).float()

    tr_mat = torch.stack((torch.stack((torch.cos(ry_diffs), torch.sin(ry_diffs), centroid_x), dim=1),
                       torch.stack((-torch.sin(ry_diffs), torch.cos(ry_diffs), centroid_z), dim=1),
                       torch.stack((zeros, zeros, ones), dim=1)),
                      dim=2)

    ones_row = torch.ones_like(x_corners)
    points_stacked = torch.stack((x_corners, z_corners, ones_row), dim=1)
    corners = torch.matmul(tr_mat, points_stacked)

    corners = corners[:, 0:2:, :]
    flat_corners = corners.contiguous().view((-1, 8))

    a = ground_plane[0]
    b = ground_plane[1]
    c = ground_plane[2]
    d = ground_plane[3]

    # Calculate heights off ground plane
    ground_y = -(a * centroid_x + c * centroid_z + d) / b
    h1 = ground_y - centroid_y
    h2 = h1 + dim_y

    batched_h1 = h1.view((-1, 1))
    batched_h2 = h2.view((-1, 1))

    # Stack into (?, 10)
    box_4c = torch.cat((flat_corners, batched_h1, batched_h2), dim=1)
    return box_4c









