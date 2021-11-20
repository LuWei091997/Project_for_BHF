import numpy as np
import cv2
import collections
from PIL import Image


def replace_color(img, dst_clr):
    '''
    通过矩阵操作颜色替换程序
    @param	img:	图像矩阵
    @param	src_clr:	需要替换的颜色(r,g,b)
    @param	dst_clr:	目标颜色		(r,g,b)
    @return				替换后的图像矩阵
    '''

    img_arr = np.asarray(img, dtype=np.double)
    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

    r_img[0 == 0] = dst_clr[0]
    g_img[0 == 0] = dst_clr[1]
    b_img[0 == 0] = dst_clr[2]
    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img


def color_dict():
    color_dict = collections.defaultdict(list)
    color_dict['orange'] = [0, 102, 255]
    color_dict['red'] = [102, 0, 255]
    color_dict['red2'] = [102, 0, 255]
    color_dict['yellow'] = [0, 255, 204]
    color_dict['green'] = [0, 255, 51]
    color_dict['cyan'] = [255, 204, 0]
    color_dict['blue'] = [255, 0, 0]
    return color_dict


def image_compose(color_arr):
    '''
    :param color_arr:
    :return:
    '''
    img = np.zeros([16, 16, 3], np.uint8)
    colors_dict = color_dict()
    image_column = len(color_arr)  # 行数
    image_row = len(color_arr[0])  # 列数
    n_image1 = []
    n_image2 = []
    for x in range(image_column):
        for y in range(image_row):
            if color_arr[x][y] in colors_dict:
                small_img = replace_color(img, colors_dict.get(color_arr[x][y]))
            else:
                small_img = replace_color(img, colors_dict.get('green'))
            n_image1.append(small_img)
        im1_ = np.concatenate(n_image1, axis=1)  # 横向拼接
        n_image1 = []
        n_image2.append(im1_)
    new_img = np.concatenate(n_image2, axis=0)  # 纵向拼接
    return new_img


if __name__ == '__main__':
    color_arr = [['blue', 'green','green'],
                 ['red', 'blue','green'],
                 ['blue', 'green', 'green']]
    im2 = image_compose(color_arr)
    cv2.imshow('', im2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im2 = Image.fromarray(im2)
    # im2.save('../output/im.jpg')
    colors_dict = color_dict()
    if 'res' in colors_dict:
        print(colors_dict)