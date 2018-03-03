"""
# ==============================
# from flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""
import png
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from PIL import Image
from .io import read_flow
from .img import flow_to_image
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

def visualize_img(img):
    """
    visualize the images using matplotlib
    :param img: input image array()
    :return: None
    """
    plt.imshow(img)
    plt.show()
    return None


def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file (.flo file)
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()

def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
        plt.imshow(img)
        plt.show()
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid
        # show
        plt.imshow(img)
        plt.show()
    return None

def segment_flow(flow):
    h = flow.shape[0]
    w = flow.shape[1]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idx = ((abs(u) > LARGEFLOW) | (abs(v) > LARGEFLOW))
    idx2 = (abs(u) == SMALLFLOW)
    class0 = (v == 0) & (u == 0)
    u[idx2] = 0.00001
    tan_value = v / u

    class1 = (tan_value < 1) & (tan_value >= 0) & (u > 0) & (v >= 0)
    class2 = (tan_value >= 1) & (u >= 0) & (v >= 0)
    class3 = (tan_value < -1) & (u <= 0) & (v >= 0)
    class4 = (tan_value < 0) & (tan_value >= -1) & (u < 0) & (v >= 0)
    class8 = (tan_value >= -1) & (tan_value < 0) & (u > 0) & (v <= 0)
    class7 = (tan_value < -1) & (u >= 0) & (v <= 0)
    class6 = (tan_value >= 1) & (u <= 0) & (v <= 0)
    class5 = (tan_value >= 0) & (tan_value < 1) & (u < 0) & (v <= 0)

    seg = np.zeros((h, w))

    seg[class1] = 1
    seg[class2] = 2
    seg[class3] = 3
    seg[class4] = 4
    seg[class5] = 5
    seg[class6] = 6
    seg[class7] = 7
    seg[class8] = 8
    seg[class0] = 0
    seg[idx] = 0

    return seg

