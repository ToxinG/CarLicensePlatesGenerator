import numpy as np
import cv2 as cv

import csv
import os

from find_corners import find_corners

old_h = 85  # 115 mm
old_w = 385  # 520 mm
new_h = 126  # 170 mm
new_w = 217  # 290 mm


def inpaint(warped):
    edge_weight = 2
    step = 3
    ext_rate = 5
    x_offset = 50  # 26
    x_search = 40
    y_top_offset = 7
    y_bottom_offset = 22
    left_edge_init = old_w // 2 - x_offset
    right_edge_init = 3 * old_w // 2 + x_offset
    # mid = (left_edge_init + right_edge_init) // 2

    # find patches with the least avg square of horizontal gradient
    g_x, _ = cv.spatialGradient(cv.cvtColor(warped, cv.COLOR_BGR2GRAY))
    g_x_sq = np.square(g_x)
    min_g_sum_l = np.sum(g_x_sq[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, left_edge_init - step: left_edge_init])
    min_g_sum_r = np.sum(g_x_sq[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, right_edge_init: right_edge_init + step])
    sum_l = min_g_sum_l
    sum_r = min_g_sum_r
    min_g_sum_l *= edge_weight
    min_g_sum_r *= edge_weight
    left_edge = left_edge_init
    right_edge = right_edge_init
    for i in range(x_search):
        weight = 2 - i / x_search

        sum_l = sum_l - np.sum(
            g_x_sq[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, left_edge_init - step + i]) + np.sum(
            g_x_sq[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, left_edge_init + i])
        if sum_l * weight <= min_g_sum_l:
            min_g_sum_l = sum_l * weight
            left_edge = left_edge_init + i + 1

        sum_r = sum_r - np.sum(
            g_x_sq[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, right_edge_init + step - i - 1]) + np.sum(
            g_x_sq[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, left_edge_init - i - 1])
        if sum_r * weight <= min_g_sum_r:
            min_g_sum_r = sum_r * weight
            right_edge = right_edge_init + step - i

    print(left_edge - left_edge_init, right_edge_init - right_edge)

    patch_left = warped[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, left_edge - step: left_edge]
    patch_right = warped[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, right_edge: right_edge + step]

    patch_left = cv.resize(patch_left, dsize=(0, 0), fx=ext_rate, fy=1)
    patch_right = cv.resize(patch_right, dsize=(0, 0), fx=ext_rate, fy=1)

    patch_left = np.concatenate((np.flip(patch_left, axis=1), patch_left), axis=1)
    patch_right = np.concatenate((patch_right, np.flip(patch_right, axis=1)), axis=1)

    # print(patch_right.shape, patch_right_flip.shape, patch_left.shape, patch_left_flip.shape)
    step *= 2 * ext_rate

    while left_edge < right_edge:
        warped[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, left_edge: left_edge + step] = patch_left
        if right_edge - left_edge > step:
            warped[2 * old_h - y_top_offset: 3 * old_h + y_bottom_offset, right_edge - step: right_edge] = patch_right
        left_edge += step
        right_edge -= step

    warped[2 * old_h - y_top_offset - 2: 2 * old_h - y_top_offset + 2,
    old_w // 2 - x_offset: 3 * old_w // 2 + x_offset] = cv.GaussianBlur(warped, ksize=(5, 5), sigmaX=0)[
                                                        2 * old_h - y_top_offset - 2: 2 * old_h - y_top_offset + 2,
                                                        old_w // 2 - x_offset: 3 * old_w // 2 + x_offset]
    warped[3 * old_h + y_bottom_offset - 2: 3 * old_h + y_bottom_offset + 2,
    old_w // 2 - x_offset: 3 * old_w // 2 + x_offset] = cv.GaussianBlur(warped, ksize=(5, 5), sigmaX=0)[
                                                        3 * old_h + y_bottom_offset - 2: 3 * old_h + y_bottom_offset + 2,
                                                        old_w // 2 - x_offset: 3 * old_w // 2 + x_offset]

    # cv.imshow('warped', warped)
    # cv.waitKey(0)

    return warped


def main():
    with open('abbyy_train.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            # try:
            filename = row['File']
            coords = [int(x) for x in row['Coords'].split(',')]
            corners = [coords[0:2], coords[2:4], coords[4:6], coords[6:8]]

            corners.sort(key=lambda x: x[0])
            left = sorted(corners[:2], key=lambda x: x[1])
            right = sorted(corners[2:], key=lambda x: x[1])
            corners = left + right

            corners_dst = [[old_w // 2, 2 * old_h], [old_w // 2, 3 * old_h],
                           [3 * old_w // 2, 2 * old_h], [3 * old_w // 2, 3 * old_h]]
            M = cv.getPerspectiveTransform(np.array(corners, dtype='float32'),
                                           np.array(corners_dst, dtype='float32'))

            img = cv.imread(os.path.join('frames_train', filename))
            warped = cv.warpPerspective(img, M, (old_w * 2, old_h * 5), borderMode=cv.BORDER_CONSTANT)

            inpainted = inpaint(warped)
            # warp_mask = np.ones(inpainted.shape[:2]) * 255
            # warp_mask =
            unwarped = cv.warpPerspective(inpainted, M, (img.shape[1], img.shape[0]), img,
                                          flags=cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_TRANSPARENT)

            # cv.imshow('unwarped', unwarped)
            # cv.waitKey(0)

            corners = cv.perspectiveTransform(np.array(corners_dst).reshape(-1,1,2).astype(np.float32), np.linalg.inv(M))

            cv.imwrite('images/inpainted/' + filename, inpainted)
            cv.imwrite('images/unwarped/' + filename, unwarped)

            i += 1
            if i % 100 == 0:
                print(i)

            # except Exception:
            #     print(row, 'error')


if __name__ == '__main__':
    main()


def old_main():
    # img = cv.imread('images/full/car_6895.jpg')
    # mask = cv.imread('images/masks/car_6895.jpg', cv.IMREAD_GRAYSCALE)
    mask_dir = 'images/masks'
    img_dir = 'images/full'
    for img_name in sorted(os.listdir(img_dir)):
        print(img_name)
        img = cv.imread(os.path.join(img_dir, img_name))
        mask = cv.imread(os.path.join(mask_dir, img_name))

        corners = find_corners(mask)
        print(corners)

        # for c in corners:
        #     print(c)
        #     print(c.shape, type(c))
        #     cv.circle(img, tuple(c), 3, (0, 255, 0), -1)

        corners_dst = [[old_w // 2, old_h], [old_w // 2, 2 * old_h], [3 * old_w // 2, old_h], [3 * old_w // 2, 2 * old_h]]
        M = cv.getPerspectiveTransform(np.array(corners, dtype='float32'), np.array(corners_dst, dtype='float32'))

        inpainted = inpaint(img, M)
        # warp_mask = np.ones(inpainted.shape[:2]) * 255
        # warp_mask =
        unwarped = cv.warpPerspective(inpainted, M, (img.shape[1], img.shape[0]), img, flags=cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_TRANSPARENT)

        # cv.imshow('unwarped', unwarped)
        # cv.waitKey(0)

        cv.imwrite('images/inpainted/' + img_name, inpainted)
        cv.imwrite('images/unwarped/' + img_name, unwarped)
