import cv2 as cv
import os
import numpy as np


def perspective_transformation(mask, new_plate):

    old_h = 96  # 115 mm
    old_w = 475  # 520 mm
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
    corners_old = [[0, 0], [0, old_h], [old_w, 0], [old_w, old_h]]

    M = cv.getPerspectiveTransform(np.array(corners_old, dtype='float32'), np.array(corners, dtype='float32'))
    warped = cv.warpPerspective(new_plate, M, (mask.shape[1], mask.shape[0]))

    # cv.imshow('result', warped)
    # cv.waitKey(0)

    return warped, corners


def gen_base(template, text):
    sign = template
    sizes = [[28, 32, 82, 74],
             [13, 79, 82, 121],
             [13, 129, 82, 171],
             [13, 179, 82, 221],
             [28, 229, 82, 271],
             [28, 276, 82, 318],
             [10, 340, 63, 373],
             [10, 380, 63, 413],
             [10, 420, 63, 453]]

    for i in range(len(text)):
        sym = text[i]
        if sym == '_':
            continue
        img = cv.imread('images/' + sym + '.png')
        img = cv.resize(img, (int(img.shape[1] * (sizes[i][2] - sizes[i][0]) / img.shape[0]), sizes[i][2] - sizes[i][0]))

        # print(img.shape)
        # cv.imshow('symbol', img)
        # cv.waitKey(0)

        sign[sizes[i][0]:sizes[i][0] + img.shape[0], sizes[i][1]:sizes[i][1] + img.shape[1]] = img

    return sign


if __name__ == '__main__':
    template = cv.imread('images/template_old.jpg')
    masks = sorted(os.listdir('images/masks'))

    for img in sorted(os.listdir('images/plates_text')):
        if os.path.exists('images/plates_text_generated/' + img):
            continue

        print(img)
        text = img.split('.')[0][-9:]
        if text[-1] == '_':
            text = text[:6] + '_' + text[-3:-1]
        new_plate = gen_base(template.copy(), text)
        mask = cv.imread('images/masks/' + img.split('.')[0][:-10] + '.jpg')
        transformed, corners = perspective_transformation(mask, new_plate)
        xs = [int(c_i[0]) for c_i in corners]
        ys = [int(c_i[1]) for c_i in corners]
        cropped = transformed[min(ys):max(ys), min(xs):max(xs), :]
        cv.imwrite('images/plates_text_generated/' + img, cropped)

    # new_plate = gen_base(template, 'M140OE_10')
    # cv.imshow('sign', new_plate)
    # cv.waitKey(0)
    # cv.imwrite('images/new_plates/car_6895.jpg', new_plate)