"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 4, suppress= True)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

import matplotlib.pyplot as plt

from src.helpers.file_io import read_json, read_numpy, read_image

list_of_heights = ["pitch0"]
list_of_towns   = ["town03"]
num_folders     = 200

def get_images(input_base_path, tzofi= False):
    folder_images = []
    # Scene level
    for height_config in list_of_heights:
        for town in list_of_towns:
            for i in range(num_folders):
                key = str(i)
                for fo in range(1):
                    img_key    =  str(fo).zfill(4) + "_00"
                    if tzofi:
                        image_path = os.path.join(input_base_path, height_config, town, key, img_key + ".jpg")
                    else:
                        image_path = os.path.join(input_base_path, height_config, town, key, "image", img_key + ".jpg")
                    folder_images.append(read_image(image_path, rgb= True))
    return folder_images


tzofi_images = get_images(input_base_path= "data/carla/splits_org/", tzofi= True)
our_images   = get_images(input_base_path= "data/carla/carla_abhinav/")

num_images = len(tzofi_images)
for i in range(num_images):
    plt.figure(dpi=200)
    plt.subplot(121)
    plt.imshow(tzofi_images[i])
    plt.subplot(122)
    plt.imshow(our_images[i])
    plt.show()
    plt.close()