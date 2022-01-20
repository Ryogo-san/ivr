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

    target_list=sorted(target_list)
    return target_list


def crop_letter(target_list, window_size,resized_size):
    for idx, target in enumerate(target_list):
        print(target)
        img = cv2.imread(target)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        label_of_target = get_label_of_path(target)

        continue_idx=0 if idx%2==0 else 26
        for y in range(h // window_size + 1):
            hiragana_unicode=hiragana_idx_to_unicode(continue_idx+y)
            out_dir = os.path.join("./dataset",label_of_target,hiragana_unicode)
            os.makedirs(out_dir,exist_ok=True)

            for x in range(w // window_size + 1):
                crop = gray[y * window_size : (y + 1) * window_size, x * window_size : (x + 1) * window_size]
                crop = np.where(crop < 210, crop, 255)

                crop_h,crop_w=crop.shape
                if crop_h==window_size and crop_w==window_size:
                    resized_crop=cv2.resize(crop,(resized_size,resized_size),interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(out_dir, f"{hiragana_unicode}_{str(x).zfill(5)}.png"), resized_crop)


def get_label_of_path(img_path):
    """ """
    dirname_list = img_path.split("/")
    return dirname_list[2]  # . tmp ball-pen r-ito.. -> ball-pen


def hiragana_idx_to_unicode(idx):
    hiragana_unicode = [
        "3042","3044","3046","3048","304A",
        "304B","304D","304F","3051","3053",
        "3055","3057","3059","305B","305D",
        "305F","3061","3064","3066","3068",
        "306A","306B","306C","306D","306E",
        "306F","3072","3075","3078","307B",
        "307E","307F","3080","3081","3082",
        "3084","3086","3088",
        "3089","308A","308B","308C","308D",
        "308F","3092","3093"
    ]
    return "U"+hiragana_unicode[idx]
  

if __name__ == "__main__":
    target_list = get_target_list("./tmp")
    crop_letter(target_list, 76,256)
