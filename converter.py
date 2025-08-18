"""
    Converts the CARLA labels to KITTI style detection labels

    Version 1 2024-02-22 Abhinav Kumar
"""
import copy
import os
import numpy as np
import datetime

from matplotlib import pyplot as plt
from plot.common_operations import *
import matplotlib
matplotlib.use( 'tkagg')

from src.helpers.file_io import read_json, read_numpy, read_image, write_lines
from src.helpers.truncation_occlusion import project_3d_points_in_4D_format, crop_boxes_in_canvas, \
    calculate_truncation, calculate_occlusion_semantics
from src.helpers.util import get_box, get_image_data, convertRot2Alpha, get_calib_text, convert_seman_ids_to_labels, \
    inch_2_meter

np.set_printoptions   (precision=2, suppress= True)

CARLA_CAMORDER = {
    'CAM_FRONT': 0,
    'CAM_FRONT_RIGHT': 1,
    'CAM_BACK_RIGHT': 2,
    'CAM_BACK': 3,
    'CAM_BACK_LEFT': 4,
    'CAM_FRONT_LEFT': 5,
}

# See https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
CARLA_SEMANTICS_MAP = {
    12: 'Pedestrian',
    13: 'Rider',
    14: 'Car',
    15: 'Truck',
    16: 'Bus',
    17: 'Train',
    18: 'Motorcycle',
    19: 'Cyclist'
}

if __name__ == '__main__':
    debug = False
    input_base_path  = "data/carla/carla_abhinav/"
    tzofi            = (input_base_path == "data/carla/splits_org")
    output_base_path = "/media/abhinav/baap2/abhinav/datasets/viewpoint/carla_kitti"
    list_of_heights  = ["pitch0",  "height6", "height12", "height18", "height24", "height27", "height30",  "height-6", "height-12", "height-18", "height-24", "height-27"]
    list_of_towns    = ["town03", "town05"]
    num_folders      = 2500

    W      = 512
    H      = 512
    pitch  = 0
    yaw    = 0
    height = 0
    cams   = ["CAM_FRONT"]
    color_list = ['r', 'b', 'k', 'pink']

    # ==============================================================================================
    # Conversion from left to right-handed coordinate system
    # ==============================================================================================
    left_to_right = np.eye(4)
    left_to_right[1, 1] = -1
    right_to_left = np.linalg.inv(left_to_right)

    # ==============================================================================================
    # Conversion from ego ground to left-handed coordinate system
    # ==============================================================================================
    ego_gd_to_carla_left = np.zeros((4, 4))
    #   Ego gd (right)           CARLA (left)
    #      Z                     Z (up)   X
    #     /                        |     /
    #    /                         |    /
    #   /                          |  /
    #  /                           |/
    # -------> X(right)            |-------> Y (right)
    # |
    # |
    # |
    # v
    # Y(down)
    #   CARLA coordinates = | 0  0  1|  | X |
    #                       | 1  0  0|  | Y |
    #                       | 0 -1  0|  | Z |
    ego_gd_to_carla_left[0, 2] =  1
    ego_gd_to_carla_left[1, 0] =  1
    ego_gd_to_carla_left[2, 1] = -1
    ego_gd_to_carla_left[3, 3] =  1
    carla_left_to_ego_gd = np.linalg.inv(ego_gd_to_carla_left)

    # All intrinsics
    nusccalib = read_json('nusccalib.json')


    # ==============================================================================================
    # Run for all height_configs and town names
    # ==============================================================================================
    #Scene level
    for height_config in list_of_heights:
        for town in list_of_towns:
            town_path = os.path.join(input_base_path, height_config, town)
            print("Running for {}".format(town_path))
            if not os.path.exists(town_path):
                continue

            # Changing origin of the 3D space
            # KITTI origin        = optical center of ego camera
            # KITTI ground origin = ground below the ego camera
            # Ego origin          = ground at the ego car 3D center.
            # These numbers can be referenced from nusccalib.json[scene_index]['CAM_FRONT']['trans']
            kitti_gd_to_kitti_cam = np.eye(4)
            kitti_gd_to_kitti_cam[1, 3] = 1.51095763913
            if height_config != 'pitch0':
                kitti_gd_to_kitti_cam[1, 3] += inch_2_meter(float(height_config.replace("height", "")))
            kitti_cam_to_kitti_gd = np.linalg.inv(kitti_gd_to_kitti_cam)

            ego_gd_to_kitti_cam    = copy.deepcopy(kitti_gd_to_kitti_cam)
            ego_gd_to_kitti_cam[0, 3] = 0.0159456324149
            ego_gd_to_kitti_cam[2, 3] = -1.70079118954
            kitti_cam_to_ego_gd = np.linalg.inv(ego_gd_to_kitti_cam)

            for i in range(num_folders):
                key = str(i)
                json_folder_path = os.path.join(town_path, key, 'info.json')
                if not os.path.exists(json_folder_path):
                    continue
                gt = read_json(json_folder_path, show_message= False)

                output_folder = os.path.join(town_path, key)
                label_folder  = os.path.join(output_folder, "label")
                calib_folder  = os.path.join(output_folder, "calib")
                # # Rename the old calib and labels
                # if os.path.exists(label_folder):
                #     os.system("mv " + label_folder + " " + label_folder + "_v0")
                # if os.path.exists(calib_folder):
                #     os.system("mv " + calib_folder + " " + calib_folder + "_v0")
                # continue
                os.makedirs(label_folder, exist_ok= True)
                os.makedirs(calib_folder, exist_ok= True)

                if not 'cam_adjust' in gt:
                   gt['cam_adjust'] = {k: {'fov': 0.0, 'yaw': 0.0} for k in CARLA_CAMORDER}
                calib = nusccalib[gt['scene_calib']]['CAM_FRONT']

                # Extrinsics are in KITTI style right-handed coordinate system.
                intrins, rots, trans = get_image_data(cams, nusccalib[gt['scene_calib']], gt['cam_adjust'], W, H, pitch, yaw, height)
                intrins_4x4 = np.eye(4)
                intrins_4x4[:3, :3] = intrins[0]
                extrins_4x4 = np.eye(4)
                extrins_4x4[:3, :3] = rots[0].transpose(1,0)
                extrins_4x4[:3,  3] = np.matmul(rots[0].transpose(1,0), -trans[0].reshape(-1, 1))[:, 0]

                # We define two p2s:
                # 1) p2_right : Projecting 3D coordinates in CARLA left-handed system to image coordinates
                # 2) p2       : Projecting 3D coordinates in KITTI coordinates to image coordinates
                p2_right = intrins_4x4 @ extrins_4x4
                p2       = p2_right @ left_to_right @ ego_gd_to_carla_left @ kitti_cam_to_ego_gd

                for fo in range(10):
                    img_key    =  str(fo).zfill(4) + "_00"
                    if tzofi:
                        image_path = os.path.join(town_path, key, img_key + ".jpg")
                    else:
                        image_path = os.path.join(town_path, key, "image", img_key + ".jpg")

                    # ==============================================================================
                    # Write calib
                    # ==============================================================================
                    output_calib_path = os.path.join(calib_folder, img_key + ".txt")
                    output_label_path = os.path.join(label_folder, img_key + ".txt")
                    calib_text = get_calib_text(p2, kitti_gd_to_kitti_cam, intrinsics= intrins_4x4[:3, :3])
                    write_lines(path= output_calib_path, lines_with_return_character= calib_text)

                    # ==============================================================================
                    # Process labels and write
                    # ==============================================================================
                    left_boxes = np.array(gt['boxes'][fo]) # N x 4 x 8
                    if left_boxes.shape[0] == 0:
                        write_lines(path= output_label_path, lines_with_return_character= [])
                        continue

                    # Project 3D to 2D
                    temp  = left_boxes.transpose(1, 0, 2) # 4 x N x 8
                    temp[1] *= -1.0
                    temp  = temp.reshape(4, -1)         # 4 X N*8
                    # temp  = np.matmul(left_to_right_matrix, temp)

                    pts2d = project_3d_points_in_4D_format(p2_right, points_4d= temp, pad_ones=False) # 4 x N*8
                    pts2d = pts2d.reshape(4, -1, 8).transpose(1, 0, 2) # N x 4 x 8

                    boxes   = [get_box(box) for box in left_boxes]
                    centers = np.array([box.center for box in boxes])
                    wlh     = np.array([box.wlh for box in boxes])
                    sin_yaw = -np.array([box.rotation_matrix[0, 0] for box in boxes])
                    cos_yaw = np.array([box.rotation_matrix[0, 1] for box in boxes])

                    num_boxes = centers.shape[0]
                    kitti_dims    = wlh[:, [2,0,1]]
                    centers_1     = np.vstack((centers.T, np.ones((1, num_boxes))))   # 4 x N
                    kitti_centers = np.matmul(ego_gd_to_kitti_cam @ carla_left_to_ego_gd, centers_1).T       # N x 4
                    kitti_yaw     = np.arctan2(sin_yaw, cos_yaw)
                    kitti_alpha   = convertRot2Alpha(kitti_yaw, z3d= kitti_centers[:, 2], x3d= kitti_centers[:, 0])

                    uncropped_bbox2d        = np.zeros((num_boxes, 4))
                    uncropped_bbox2d[:, 0]  = np.min(pts2d[:, 0], axis= 1)
                    uncropped_bbox2d[:, 1]  = np.min(pts2d[:, 1], axis= 1)
                    uncropped_bbox2d[:, 2]  = np.max(pts2d[:, 0], axis= 1)
                    uncropped_bbox2d[:, 3]  = np.max(pts2d[:, 1], axis= 1)
                    kitti_bbox2d            = crop_boxes_in_canvas(cam_bboxes= uncropped_bbox2d)
                    kitti_truncation        = calculate_truncation(uncropped_bbox= uncropped_bbox2d, cropped_bbox= kitti_bbox2d)

                    if tzofi:
                        kitti_occlusion  = np.zeros((num_boxes, ))
                        kitti_seman      = 14*np.ones((num_boxes, ))
                    else:
                        depth_path = os.path.join(town_path, key, "depth", img_key + ".npy")
                        seman_path = os.path.join(town_path, key, "seman", img_key + ".npy")
                        depth = read_numpy(depth_path, show_message= False)
                        seman = read_numpy(seman_path, show_message= False)

                        kitti_occlusion, kitti_seman = calculate_occlusion_semantics(bbox_2d= kitti_bbox2d, centers= kitti_centers, hlw= kitti_dims, seman_map= seman, depth_map= depth)

                    kitti_labels = convert_seman_ids_to_labels(seman_ids= kitti_seman, mapping= CARLA_SEMANTICS_MAP)
                    kitti_centers_bot = copy.deepcopy(kitti_centers)
                    kitti_centers_bot[:, 1] += kitti_dims[:, 0]/2.0

                    label_text = ""
                    for b in range(num_boxes):
                        label_text += kitti_labels[b]
                        label_text += " {:.2f} {:d}".format(kitti_truncation[b], kitti_occlusion[b])
                        label_text += " {:.2f}".format(kitti_alpha[b])
                        label_text += " {:.2f} {:.2f} {:.2f} {:.2f}".format(kitti_bbox2d[b, 0], kitti_bbox2d[b, 1], kitti_bbox2d[b, 2], kitti_bbox2d[b, 3])
                        label_text += " {:.2f} {:.2f} {:.2f}".format(kitti_dims[b, 0], kitti_dims[b, 1], kitti_dims[b, 2])
                        label_text += " {:.2f} {:.2f} {:.2f}".format(kitti_centers_bot[b, 0], kitti_centers_bot[b, 1], kitti_centers_bot[b, 2])
                        label_text += " {:.2f}\n".format(kitti_yaw[b])
                    write_lines(path= output_label_path, lines_with_return_character= label_text)

                    if debug:
                        kitti_pcs = project_3d_points_in_4D_format(p2, kitti_centers.T, pad_ones= False).T # N x 2
                        image = read_image(image_path, rgb=True)
                        plt.imshow(image)
                        plt.title(key)
                        ax = plt.gca()

                        for id in range(pts2d.shape[0]):
                            for v in range(8):
                                draw_circle(ax, x= pts2d[id, 0, v], y= pts2d[id, 1, v], radius=3, color= color_list[id % 4])
                            draw_circle(ax, x= kitti_pcs[id, 0], y= kitti_pcs[id, 1], color= color_list[id % 4])
                        plt.show()

                if (i+1) % 250 == 0 or (i+1) == num_folders:
                    e = datetime.datetime.now()
                    print(e.strftime("[%Y-%m-%d %H:%M:%S] ") + "{:4d} folders done.".format(i+1))
