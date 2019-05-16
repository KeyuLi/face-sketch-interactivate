import numpy as np
import cv2
import os


def get_data_list(data_dir):
    photo_dir = data_dir + 'photo/'
    sketch_dir = data_dir + 'sketch/'
    photo_list = np.sort(os.listdir(photo_dir))
    sketch_list = np.sort(os.listdir(sketch_dir))
    photo_data_list = np.zeros((len(photo_list), 250, 200, 3))
    sketch_data_list = np.zeros((len(sketch_list), 250, 200, 3))
    for i, image_name in enumerate(photo_list):
        image_photo_dir = photo_dir + image_name
        photo_data = cv2.imread(image_photo_dir)
        photo_data_list[i, :, :, :] = photo_data
    for i, image_name in enumerate(sketch_list):
        image_sketch_dir = sketch_dir + image_name
        sketch_data = cv2.imread(image_sketch_dir)
        sketch_data_list[i, :, :, :] = sketch_data
    return photo_data_list, sketch_data_list
