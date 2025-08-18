import os, sys
sys.path.append(os.getcwd())

import numpy as np
import math
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

def get_box(box):
    diffw = box[:3, 1] - box[:3, 2]
    diffl = box[:3, 0] - box[:3, 1]
    diffh = box[:3, 4] - box[:3, 0]

    center = (box[:3, 4] + box[:3, 2]) / 2
    # carla flips y axis
    center[1] = -center[1]

    dims = [np.linalg.norm(diffw), np.linalg.norm(diffl), np.linalg.norm(diffh)]

    EPS = 1e-2
    if dims[0] < EPS or dims[1] < EPS or dims[2] < EPS:
        rot = np.eye(3)
    else:
        rot = np.zeros((3, 3))
        rot[:, 1] = diffw / dims[0]
        rot[:, 0] = diffl / dims[1]
        rot[:, 2] = diffh / dims[2]

        # quat = Quaternion(matrix=rot)
        # again, carla flips y axis
        # newquat = Quaternion(quat.w, -quat.x, quat.y, -quat.z)
        # again, carla flips y axis
        # See https://stackoverflow.com/a/38124709
        rot[:, 1] *= -1
    quat = Quaternion(matrix=rot)
    newquat = quat

    nbox = Box(center, dims, newquat)
    return nbox

def get_kitti_box(box):
    diffw = box[:3, 1] - box[:3, 2]
    diffl = box[:3, 0] - box[:3, 1]
    diffh = box[:3, 4] - box[:3, 0]

    center = (box[:3, 4] + box[:3, 2]) / 2

    dims = [np.linalg.norm(diffw), np.linalg.norm(diffl), np.linalg.norm(diffh)]

    EPS = 1e-2
    if dims[0] < EPS or dims[1] < EPS or dims[2] < EPS:
        rot = np.eye(3)
    else:
        rot = np.zeros((3, 3))
        rot[:, 1] = diffw / dims[0]
        rot[:, 0] = diffl / dims[1]
        rot[:, 2] = diffh / dims[2]

        # quat = Quaternion(matrix=rot)
        # again, carla flips y axis
        # newquat = Quaternion(quat.w, -quat.x, quat.y, -quat.z)
        # again, carla flips y axis
        # See https://stackoverflow.com/a/38124709
        rot[:, 1] *= -1
    quat = Quaternion(matrix=rot)
    newquat = quat

    nbox = Box(center, dims, newquat)
    return nbox


def get_ixes(split="train"):
    # if split == "train":
    folders = np.arange(2500)
    files = np.arange(10)
    ixes = []

    total_files = folders.shape[0] * files.shape[0]
    for f in folders:
        for f0 in files:
            ixes.append((f, f0))

    return ixes


def parse_calibration(calib, width, height, cam_adjust, pitch=0, yaw=0, height_adjust=0):
    info = {}
    for k, cal in calib.items():
        if k == 'LIDAR_TOP':
            continue
        if pitch == 0:
            pitch = cam_adjust[k]['pitch']
        if yaw == 0:
            yaw = cam_adjust[k]['yaw']
        if height_adjust == 0:
            height_adjust = cam_adjust[k]['height']

        trans = [cal['trans'][0], -cal['trans'][1], cal['trans'][2] + height_adjust]
        intrins = np.identity(3)
        intrins[0, 2] = width / 2.0
        intrins[1, 2] = height / 2.0
        intrins[0, 0] = intrins[1, 1] = width / (
                    2.0 * np.tan((cal['fov'] + cam_adjust[k]['fov']) * np.pi / 360.0))

        coordmat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        rot = np.matmul(
            Quaternion(axis=[0, 0, 1], angle=np.radians(-(cal['yaw'] + yaw))).rotation_matrix,
            np.linalg.inv(coordmat))
        if pitch != 0:
            rot = np.matmul(
                Quaternion(axis=[0, 1, 0], angle=np.radians(-(pitch))).rotation_matrix,
                np.linalg.inv(coordmat))

        quat = Quaternion(matrix=rot)

        info[k] = {'trans': trans, 'intrins': intrins, 'rot': quat}

    return info


def get_image_data(cams, calib, cam_adjust, W, H, pitch, yaw, height):
    intrins = []
    rots = []
    trans = []
    cal = parse_calibration(calib, width=W, height=H,
                            cam_adjust=cam_adjust, pitch=pitch, yaw=yaw, height_adjust=height)
    for cam in cams:
        intrin = np.array(cal[cam]['intrins'])
        rot = np.array(cal[cam]['rot'].rotation_matrix)
        tran = np.array(cal[cam]['trans'])

        intrins.append(intrin)
        rots.append(rot)
        trans.append(tran)

    return (np.stack(intrins), np.stack(rots), np.stack(trans))

def convertRot2Alpha(ry3d, z3d, x3d):

    if type(z3d) == np.ndarray:
        alpha = ry3d - np.arctan2(-z3d, x3d) - 0.5 * math.pi

        while np.any(alpha > math.pi): alpha[alpha > math.pi] -= math.pi * 2
        while np.any(alpha <= (-math.pi)): alpha[alpha <= (-math.pi)] += math.pi * 2
    else:
        alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
        #alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi

        while alpha > math.pi: alpha -= math.pi * 2
        while alpha <= (-math.pi): alpha += math.pi * 2

    return alpha

def project_3d_points_in_4D_format(p2, points_4d, pad_ones= False):
    """
    Projects 3d points appened with ones to 2d using projection matrix
    :param p2:       np array 4 x 4
    :param points:   np array 4 x N
    :return: coord2d np array 4 x N
    """
    N = points_4d.shape[1]
    z_eps = 1e-2

    if type(points_4d) == np.ndarray:
        if pad_ones:
            points_4d = np.vstack((points_4d, np.ones((1, N))))

        coord2d = np.matmul(p2, points_4d)
        ind = np.where(np.abs(coord2d[2]) > z_eps)

    coord2d[:2, ind] /= coord2d[2, ind]

    return coord2d

def inch_2_meter(inches):
    return inches * 0.0254

def convert_seman_ids_to_labels(seman_ids, mapping):
    output = []
    for id in seman_ids:
        if id in mapping.keys():
            output.append(mapping[id])
        else:
            output.append('DontCare')

    return output

def format_one_matrix(t):
    def float_formatter(x):
        return f"{x:.12e}"
    t = t[:3, :].flatten()
    return np.array2string(t, formatter={'float_kind': float_formatter}).replace("\n", "").replace("[", "").replace("]", "")

def get_calib_text(p2, gd_to_cam= np.eye(4), intrinsics= np.eye(3)):
    identity = np.eye(4)
    identity_3 = np.eye(3)

    calib_text = []
    # calib_text += "P0: " + format_one_matrix(identity) + "\n"
    # calib_text += "P1: " + format_one_matrix(identity) + "\n"
    # calib_text += "P2: " + format_one_matrix(p2) + "\n"
    # calib_text += "R0_rect: " + format_one_matrix(identity) + "\n"
    # calib_text += "Tr_velo_to_cam: " + format_one_matrix(identity) + "\n"
    # calib_text += "Tr_imu_to_velo: " + format_one_matrix(identity) + "\n"

    calib_text.append("P0: " + format_one_matrix(identity) + "\n")
    calib_text.append("P1: " + format_one_matrix(identity) + "\n")
    calib_text.append("P2: " + format_one_matrix(p2) + "\n")
    calib_text.append("P3: " + format_one_matrix(identity) + "\n")
    calib_text.append("R0_rect: " + format_one_matrix(identity_3) + "\n")
    calib_text.append("Tr_velo_to_cam: " + format_one_matrix(identity) + "\n")
    calib_text.append("Tr_imu_to_velo: " + format_one_matrix(identity) + "\n")
    calib_text.append("gd_to_cam: " + format_one_matrix(gd_to_cam) + "\n")
    calib_text.append("intrinsics: " + format_one_matrix(intrinsics) + "\n")

    return calib_text