import cv2
import pytesseract

custom_config = r'-c tessedit_char_whitelist=0123456789 --psm 6'


def get_num(color, img):
    '''
    :param color: ['red', 'orange', 'yellow', 'green', 'cyan-blue', 'blue']
    :param img:
    :return:
    '''
    dis = 39
    i = 0
    # 控制X的数值，用于改变读取的位置
    if color == 'red':
        i = 0
    elif color == 'orange':
        i = 1
    elif color == 'yellow':
        i = 2
    elif color == 'green':
        i = 3

    elif color == 'cyan-blue':
        i = 6
    elif color == 'blue':
        i = 7
    position_up = dis * i + 86
    position_down = dis * i + 118
    position_up = int(position_up)
    position_down = int(position_down)

    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_1 = cv2.inRange(img2[position_up:position_down, 95:117], lowerb=160, upperb=200)  # 控制图像大小，并清除背景
    img_2 = cv2.inRange(img2[position_up:position_down, 124:144], lowerb=160, upperb=200)
    img_3 = cv2.inRange(img2[position_up:position_down, 144:162], lowerb=160, upperb=200)
    img_4 = cv2.inRange(img2[position_up:position_down, 160:182], lowerb=160, upperb=200)

    '''
    cv2.imshow('1', img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('1', img_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('1', img_3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('1', img_4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    num_1 = pytesseract.image_to_string(img_1, config=custom_config)
    num_2 = pytesseract.image_to_string(img_2, config=custom_config)
    num_3 = pytesseract.image_to_string(img_3, config=custom_config)
    num_4 = pytesseract.image_to_string(img_4, config=custom_config)
    # print(num_1[:1], num_2[:1], num_3[:1], num_4[:1])
    num_ = round(
        int(num_1[:1]) + 0.1 * int(num_2[:1]) + 0.01 * int(num_3[:1]) + 0.001 * int(num_4[:1]), 3)
    print(num_)
    return num_


if __name__ == '__main__':
    path = '../data/picture/1/12.PNG'
    img = cv2.imread(path)
    color = ['red', 'orange', 'yellow', 'green', 'cyan-blue', 'blue']
    for i in color:
        get_num(i, img)
