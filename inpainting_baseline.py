import numpy as np
import cv2 as cv


def inpaint(img, mask):
    res = cv.inpaint(img, mask, 129, cv.INPAINT_NS)
    cv.imshow('result', res)
    cv.waitKey(800)
    return res


def extend_mask(mask, ext_radius=4):
    new_mask = mask.copy()
    n = mask.shape[0]
    m = mask.shape[1]
    for i in range(n):
        for j in range(m):
            if mask[i][j]:
                for ki in range(max(0, i - ext_radius), min(n, i + ext_radius + 1)):
                    for kj in range(max(0, j - ext_radius), min(m, j + ext_radius + 1)):
                        new_mask[ki][kj] = True
    return new_mask


if __name__ == '__main__':
    img = cv.imread('images/full/car_6895.jpg')
    mask = cv.imread('images/masks/car_6895.jpg', cv.CV_8UC1)
    mask -= 255
    # mask = extend_mask(mask, ext_radius=7)
    res = inpaint(img, mask)
    cv.imwrite('images/inpainted/baseline_demo.jpg', res)
