import cv2 as cv
import numpy as np
import os

import find_corners


def gen_base(template, text):
    sign = template
    sizes = [[10, 36, 45, 31],
             [10, 89, 45, 28],
             [10, 122, 45, 28],
             [10, 155, 45, 28],
             [70, 23, 45, 31],
             [70, 63, 45, 31],
             [69, 128, 35, 17],
             [69, 152, 35, 22],
             [69, 178, 35, 22]]

    for i in range(len(text)):
        sym = text[i]
        if sym == '_':
            continue
        img = cv.imread('images/' + sym + '.png')
        img = cv.resize(img, (int(img.shape[1] * sizes[i][2] / img.shape[0]), sizes[i][2]))

        # print(img.shape)
        # cv.imshow('symbol', img)
        # cv.waitKey(0)

        sign[sizes[i][0]:sizes[i][0] + img.shape[0], sizes[i][1]:sizes[i][1] + img.shape[1]] = img

    return sign


if __name__ == '__main__':
    template = cv.imread('images/template.png')
    template_new_mask = cv.imread('images/template_new_mask.png')
    masks = sorted(os.listdir('images/masks'))

    for img in sorted(os.listdir('images/plates_text')):
        # if os.path.exists('images/rendered_new/' + img):
        #     continue
        #
        # print(img)

        text = img.split('.')[0][-9:]
        if text[-1] == '_':
            text = text[:6] + '_' + text[-3:-1]
        new_plate = gen_base(template.copy(), text)
        mask = cv.imread('images/masks/' + img.split('.')[0][:-10] + '.jpg')
        corners = find_corners.find_corners(mask)

        p_t = cv.getPerspectiveTransform(np.array(find_corners.corners_old, dtype='float32'),
                                         np.array(corners, dtype='float32'))
        transformed = cv.warpPerspective(new_plate, p_t, (mask.shape[1], mask.shape[0]))
        new_mask = cv.warpPerspective(template_new_mask, p_t, (mask.shape[1], mask.shape[0]))
        # xs = [int(c_i[0]) for c_i in corners]
        # ys = [int(c_i[1]) for c_i in corners]
        # cropped = transformed[min(ys):max(ys), min(xs):max(xs), :]
        cv.imwrite('images/rendered_new/' + img, transformed)
        cv.imwrite('images/new_masks/' + img, new_mask)
