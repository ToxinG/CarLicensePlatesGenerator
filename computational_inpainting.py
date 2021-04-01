import numpy as np
import cv2 as cv

import os

from perspective_transformation import perspective_transformation

old_h = 85  # 115 mm
old_w = 385  # 520 mm
new_h = 126  # 170 mm
new_w = 217  # 290 mm


def inpaint(img, p_t):
    inverse_p_t = np.linalg.inv(p_t)
    warped = cv.warpPerspective(img, p_t, (old_w * 2, old_h * 3), borderMode=cv.BORDER_CONSTANT)

    step = 5
    ext_rate = 4
    offset = 20
    left_edge = old_w // 2 - offset
    right_edge = 3 * old_w // 2 + offset
    patch_left = warped[old_h - 5: 2 * old_h + 5, left_edge - step: left_edge]
    patch_right = warped[old_h - 5: 2 * old_h + 5, right_edge: right_edge + step]

    patch_left = cv.resize(patch_left, dsize=(0, 0), fx=ext_rate, fy=1)
    patch_right = cv.resize(patch_right, dsize=(0, 0), fx=ext_rate, fy=1)

    patch_left_flip = np.flip(patch_left, axis=1)
    patch_right_flip = np.flip(patch_right, axis=1)

    print(patch_right.shape, patch_right_flip.shape, patch_left.shape, patch_left_flip.shape)
    step *= ext_rate

    while left_edge < right_edge:
        warped[old_h - 5: 2 * old_h + 5, left_edge: left_edge + step] = patch_left_flip
        warped[old_h - 5: 2 * old_h + 5, right_edge - step: right_edge] = patch_right_flip
        left_edge += step
        right_edge -= step
        warped[old_h - 5: 2 * old_h + 5, left_edge: left_edge + step] = patch_left
        warped[old_h - 5: 2 * old_h + 5, right_edge - step: right_edge] = patch_right
        left_edge += step
        right_edge -= step

    # cv.imshow('warped', warped)
    # cv.waitKey(0)

    cv.imwrite('images/inpainted/comp_inpainted.jpg', warped)

    return warped


if __name__ == '__main__':

    # img = cv.imread('images/full/car_6895.jpg')
    # mask = cv.imread('images/masks/car_6895.jpg', cv.IMREAD_GRAYSCALE)
    mask_dir = 'images/masks'
    img_dir = 'images/full'
    for img_name in sorted(os.listdir(img_dir)):
        print(img_name)
        img = cv.imread(os.path.join(img_dir, img_name))
        mask = cv.imread(os.path.join(mask_dir, img_name))

        corners, p_t = perspective_transformation(mask)
        print(corners)

        # for c in corners:
        #     print(c)
        #     print(c.shape, type(c))
        #     cv.circle(img, tuple(c), 3, (0, 255, 0), -1)

        corners_dst = [[old_w // 2, old_h], [old_w // 2, 2 * old_h], [3 * old_w // 2, old_h], [3 * old_w // 2, 2 * old_h]]
        M = cv.getPerspectiveTransform(np.array(corners, dtype='float32'), np.array(corners_dst, dtype='float32'))

        inpainted = inpaint(img, M)

        cv.imwrite('images/inpainted/' + img_name, inpainted)
