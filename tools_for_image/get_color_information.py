import numpy as np


def get_info(mask, img, x1, x2, y1, y2):
    colored_img = np.zeros((len(mask), len(mask[0]), 3))
    color_info = []
    for i in range(len(mask)):
        if x1 < i < x2:
            for j in range(len(mask[i])):
                if y1 < j < y2:
                    if mask[i][j] != 0:
                        # 获取真实颜色
                        colored_img[i][j] = img[i, j]
                        # 确定该像素位置
                        color_info.append([i, j])
    return color_info, colored_img
