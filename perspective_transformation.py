import cv2 as cv
import numpy as np
import os


def perspective_transformation(mask, new_plate):

    old_h = 85  # 115 mm
    old_w = 385  # 520 mm
    new_h = 126  # 170 mm
    new_w = 217  # 290 mm

    gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    canny_output = cv.Canny(gray, 50, 150, apertureSize=3)
    _, contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)

    # for i in range(len(contours)):
    #     cv.drawContours(img, contours, i, (255, 0, 0))
    black = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    black = cv.drawContours(black, hull_list, 0, (255, 255, 255))

    # cv.imshow('drawing', black)
    # cv.waitKey(0)

    black = cv.cvtColor(black, cv.COLOR_BGR2GRAY)
    canny_output = cv.Canny(black, 50, 150, apertureSize=3)
    corners = cv.goodFeaturesToTrack(canny_output, 4, 0.01, 10)
    corners = [c[0] for c in corners]
    corners.sort(key=lambda x: x[0])
    left = sorted(corners[:2], key=lambda x: x[1])
    right = sorted(corners[2:], key=lambda x: x[1])
    corners = left + right
    # corners = [[c_i[0] - corners[0][0], c_i[1] - corners[0][1]] for c_i in corners]
    corners_old = [[(new_w - old_w) / 2, (new_h - old_h) / 2], [(new_w - old_w) / 2, (new_h + old_h) / 2],
                   [(new_w + old_w) / 2, (new_h - old_h) / 2], [(new_w + old_w) / 2, (new_h + old_h) / 2]]

    M = cv.getPerspectiveTransform(np.array(corners_old, dtype='float32'), np.array(corners, dtype='float32'))
    warped = cv.warpPerspective(new_plate, M, (mask.shape[1], mask.shape[0]), borderMode=cv.BORDER_TRANSPARENT)

    # cv.imshow('result', warped)
    # cv.waitKey(0)

    return warped, corners

    # for c in corners:
    #     print(c)
    #     cv.circle(black, tuple(c), 3, 255, -1)
    # cv.imshow('corners', black)
    # cv.waitKey(0)


if __name__ == '__main__':
    img_dir = 'images/plates_text'
    for img_name in os.listdir(img_dir):
        print(img_name)
        mask = cv.imread(os.path.join(img_dir, img_name))
        new_plate = cv.imread('images/new_plates/car_6895.jpg')
        cv.imwrite(os.path.join('images/results', img_name), perspective_transformation(mask, new_plate)[0])
