import cv2
import numpy as np


def extract_red(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 区间1
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 区间2
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 拼接两个区间
    mask = mask0 + mask1
    return mask


def extract_yellow(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([26, 43, 46])
    upper_red = np.array([34, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    return mask


def extract_orange(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([11, 43, 46])
    upper_red = np.array([25, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    return mask


def extract_blue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([100, 120, 46])
    upper_red = np.array([124, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    return mask


def extract_white(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 221])
    upper_red = np.array([180, 30, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    return mask


def extract_blue_for_empty(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([100, 43, 46])
    upper_red = np.array([124, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    return mask


def extract_gray(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 46])
    upper_red = np.array([180, 43, 220])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    return mask


def extract_cyan_blue(img):  # 青色
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([78, 43, 46])
    upper_red = np.array([99, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    return mask
