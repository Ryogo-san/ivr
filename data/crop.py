import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_target_list(tmp_dir):
    """ """
    target_list = []
    for dirname, _, filenames in os.walk(tmp_dir):
        for filename in filenames:
            png_file = os.path.join(dirname, filename)
            if filename[-4:] == ".png":
                target_list.append(png_file)

    return target_list


def crop_letter(target_list, window_size):
    for idx, target in enumerate(target_list):
        img = cv2.imread(target)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        label_of_target = get_label_of_path(target)
        out_dir = "./dataset/" + label_of_target

        for y in range(h // window_size + 1):
            for x in range(w // window_size + 1):
                crop = gray[y * window_size : (y + 1) * window_size, x * window_size : (x + 1) * window_size]
                crop = np.where(crop < 210, crop, 255)
                plt.imshow(crop, cmap="gray")
                plt.axis("off")
                cv2.imwrite(os.path.join(out_dir, f"{idx+1}_{y+1}_{x+1}.png"), crop)


def get_label_of_path(img_path):
    """ """
    dirname_list = img_path.split("/")
    return dirname_list[2]  # . tmp ball-pen r-ito.. -> ball-pen


if __name__ == "__main__":
    target_list = get_target_list("./tmp")
    crop_letter(target_list, 76)
