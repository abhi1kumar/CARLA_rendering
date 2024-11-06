"""
    Sample Run:
    python .py
"""
import os, sys

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import numpy as np

np.set_printoptions   (precision= 4, suppress= True)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
import cv2

from matplotlib.patches import Circle

from plot.file_io import read_image, read_numpy

def map_image_via_pallete(image, my_pallete):
    num_keys = len(my_pallete.keys())
    r, c = image.shape
    output = np.zeros((r, c, 3), dtype= np.uint8)
    for i in range(num_keys):
        mask = image == i
        output[mask] = my_pallete[i]

    return output

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

vmin = 0
vmax = 50

cmap = matplotlib.cm.get_cmap('magma_r')
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

c = 5
ilist = [4]#sorted(np.random.choice(17, 17, replace= False))
flist = [0] * len(ilist)
town = 'town03'

ilist = [1054]#, 1021, 1033, 1061, 1077, 1114, 1209, 1278]#sorted(np.random.choice(1500, 200, replace= False))
flist = [9]#, 5, 3, 5, 3, 1, 6, 0]
town = 'town05'

heights_list = ['height-27', 'height-24', 'height-18', 'height-12', 'height-6', 'height0', 'height6', 'height12', 'height18', 'height24', 'height30']

P2 = np.eye(4)
P2[0, 2] = 256.0
P2[1, 2] = 256.0
P2[0, 0] = 405.24
P2[1, 1] = 405.24

for height in heights_list:
    folder = os.path.join('data/carla/carla_abhinav/', height, town)

    for i, f in zip(ilist, flist):
        key = str(i)
        f1 = str(f).zfill(4)
        image_path  = os.path.join(folder, key, 'image', f1 + '_00.jpg')
        depth_path  = os.path.join(folder, key, 'depth', f1 + '_00.npy')
        seman_path  = os.path.join(folder, key, 'seman', f1 + '_00.npy')
        seman_path2 = os.path.join(folder, key, 'seman', f1 + '_00.jpg')
        seman_path3 = os.path.join(folder, key, 'seman', f1 + '_00.png')
        lidar_path  = os.path.join(folder, key, 'lidar', f1 + '_00.npy')

        # print(image_path)
        image = read_image(image_path, rgb= True)
        depth = read_numpy(depth_path)
        seman = read_numpy(seman_path)
        lidar = read_numpy(lidar_path)

        coord2d = project_3d_points_in_4D_format(p2= P2, points_4d= lidar[:, :3].transpose(), pad_ones= True).transpose() #[N, 4]
        idx = np.random.choice(np.arange(coord2d.shape[0]), 20)
        mask = coord2d[:, 2] > 0
        coord2d = coord2d[mask]
        mask = coord2d[:, 0] > 0
        coord2d = coord2d[mask]
        mask = coord2d[:, 0] < 512
        coord2d = coord2d[mask]
        mask = coord2d[:, 1] > 0
        coord2d = coord2d[mask]
        mask = coord2d[:, 1] < 512
        coord2d = coord2d[mask]
        # print(lidar[idx])
        # print(coord2d.shape)

        if os.path.exists(seman_path2) and os.path.exists(seman_path3):
            c = 6
            diff = 2
            seman2 = cv2.imread(seman_path2, cv2.IMREAD_GRAYSCALE)
            seman3 = cv2.imread(seman_path2, cv2.IMREAD_GRAYSCALE)
        else:
            c = 4
            diff = 0

        plt.figure(figsize= (24,6), dpi= params.DPI)
        plt.subplot(1,c,1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Image'.format(key))
        plt.subplot(1,c,2)
        plt.imshow(map_image_via_pallete(seman, params.cityscapes_color_palette))
        plt.axis('off')
        if os.path.exists(seman_path2) and os.path.exists(seman_path3):
            plt.title('Seman NPY')
            plt.subplot(1,c,3)
            plt.imshow(map_image_via_pallete(seman3, params.cityscapes_color_palette))
            plt.axis('off')
            plt.title('Seman PNG')
            plt.subplot(1,c,4)
            plt.imshow(map_image_via_pallete(seman2, params.cityscapes_color_palette))
            plt.axis('off')
            plt.title('Seman JPG')
        else:
            plt.title('Seman')
        plt.subplot(1,c,3+diff)
        plt.imshow(depth, cmap= 'magma_r', vmin= 0, vmax= vmax)
        plt.axis('off')
        plt.title('Depth')
        plt.subplot(1,c,4+diff)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Lidar')
        ax = plt.gca()
        for j in range(coord2d.shape[0]):
            ax.add_patch(
                Circle((coord2d[j, 0], coord2d[j, 1]), radius=1, color=cmap(norm(coord2d[j, 2]))))


        out_folder = os.path.join("images/gt", town, key)
        os.makedirs(out_folder, exist_ok= True)
        savefig(plt, os.path.join(out_folder, height + "_{}_".format(key) + f1 + "_seman_depth_lidar.png"), newline= False)
        # plt.show()
        plt.close()
