import cv2
import numpy as np
from tools_for_image import get_the_number
from normal_tools import read_data
from tools_for_image import read_all_pictures
from tools_for_image import get_color_list
from tools_for_image import creat_similar_image
import time

np.set_printoptions(threshold=np.inf)


def get_color(frame):
    '''
    :param frame: 图片
    :return: 图片中颜色占比最多
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = get_color_list.getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        # cv2.imwrite(d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]  # 返回阈值化后的图像
        binary = cv2.dilate(binary, None, iterations=2)  # 膨胀处理
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d
    return color


def split_picture(img):
    '''
    :param img: entire picture
    :return: cut into 16 pieces
    '''
    # 裁剪
    img = img[60:860, 500:1300]
    vague_img = []
    step = 16
    range_1 = int(800 / step)  # must be integer 50
    for i in range(0, 800, step):
        for j in range(0, 800, step):
            # print(i, j)
            img_part = img[i:i + step, j:j + step]
            vague_img.append(img_part)
    final_img = []
    for h in range(0, range_1):
        for j in range(0, range_1):
            if j == 0:
                final_img.append([])
            # print(4 * h + j)
            final_img[h].append(vague_img[range_1 * h + j])
        # print('time', h)
    return final_img


def get_color_(image, thicken_red, thicken_blue, thicken_green, thicken_yellow, thicken_cyan_blue, thicken_orange):
    pic = []
    pic_w_col = []
    sa_color = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            color = get_color(image[i][j])
            # print(color)
            sa_color.append(color)
            # print(color)
            if color == 'red':  # 红色    需要进行空白识别及处理
                pic.append([i, j, thicken_red])
            elif color == 'red2':  # 红色
                pic.append([i, j, thicken_red])
            elif color == 'blue':  # 蓝色
                pic.append([i, j, thicken_blue])
            elif color == 'cyan_white':  # 青色_白
                pic.append([i, j, thicken_green])
            elif color == 'cyan':  # 青色
                pic.append([i, j, thicken_cyan_blue])
            elif color == 'yellow':  # 黄色
                pic.append([i, j, thicken_yellow])
            elif color == 'orange':  # 橙色
                pic.append([i, j, thicken_orange])
            else:
                pic.append([i, j, 0])
        pic_w_col.append(sa_color)
        sa_color = []
    return pic, pic_w_col


def insert_param(data, Parameters):
    _param = []
    data_ = []
    for i in data:
        _param.extend(i[0:2])
        _param.extend(Parameters)
        _param.extend(i[2:])
        data_.append(_param)
        _param = []
    return data_


if __name__ == '__main__':
    path_ = '../data/picture'
    all_path = read_all_pictures.read_pictures(path_)
    param_path = '../data/全部数据.csv'
    param = read_data.readfile(param_path)
    param = param[:, 0:-2]
    final_data = []
    for path in all_path:
        img = cv2.imread(path)
        print('第', path, '次')
        start_time = time.time()
        # 获取是哪个图片，与工艺参数对应
        a = path.split('/', )
        b = a[4].split('.')
        num = (int(a[3]) - 1) * 100 + int(b[0])
        parameters = param[num - 1]
        # get thicken info from picture
        thicken_red = get_the_number.get_num('red', img)  # 获取相应厚度
        thicken_red = round((thicken_red - 1) * 100, 2)

        thicken_blue = get_the_number.get_num('blue', img)  # 获取相应厚度
        thicken_blue = round((thicken_blue - 1) * 100, 2)

        thicken_green = get_the_number.get_num('green', img)  # 获取相应厚度
        thicken_green = round((thicken_green - 1) * 100, 2)

        thicken_yellow = get_the_number.get_num('yellow', img)  # 获取相应厚度
        thicken_yellow = round((thicken_yellow - 1) * 100, 2)

        thicken_cyan_blue = get_the_number.get_num('cyan-blue', img)  # 获取相应厚度
        thicken_cyan_blue = round((thicken_cyan_blue - 1) * 100, 2)

        thicken_orange = get_the_number.get_num('orange', img)  # 获取相应厚度
        thicken_orange = round((thicken_orange - 1) * 100, 2)
        # split picture 80*80
        image = split_picture(img)
        # insert param
        data_with_thicken, color_name = get_color_(image, thicken_red, thicken_blue, thicken_green, thicken_yellow,
                                                   thicken_cyan_blue,
                                                   thicken_orange)
        # print(color_name)
        sim_img = creat_similar_image.image_compose(color_name)
        path_im = '../output/sim_picture/' + str(num - 1) + '.jpg'
        cv2.imwrite(path_im, sim_img)
        data = insert_param(data_with_thicken, parameters)  # 输入一个图形的所有位置信息以及厚度，加工参数
        final_data.append(data)
        # print(data)
        end_time = time.time()
        print('所用时间：%fs' % (end_time - start_time))
        del img
        del start_time
        del end_time
        del sim_img
    final_data = np.array(final_data)
    np.save(file='../output/final_data_green_0.npy', arr=final_data)
    print('finsh')
    # 输出的矩阵尺寸为range_1*range_1*51
    # 输入尺寸为range_1*range_1*50
    # lable range_1*range_1*1
    '''from playsound import playsound
    path = '../sound/8517.wav'
    playsound(path)'''

