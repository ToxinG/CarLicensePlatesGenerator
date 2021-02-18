import math

import numpy as np
import cv2 as cv


def inpaint(img, mask):
    kernel_size = 9
    kernel_r = (kernel_size - 1) // 2
    area = kernel_size * kernel_size
    h, w = img.shape[0:2]
    img_8uc1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    confidence: np.ndarray = mask.astype(dtype='int')
    confidence //= confidence.max(initial=0)
    img[confidence == 0] = [0, 0, 0]

    # for proper contour finding, to have the object black, the background white
    # and the contour (contours[0], there are 2 of them) as pixels of the object itself, not outside:
    mask -= 255
    mask *= -1

    gx, gy = cv.spatialGradient(img_8uc1, ksize=kernel_size)
    g = np.empty(shape=img.shape[0:2])
    g_0 = np.sqrt(np.add(np.square(gx), np.square(gy)))
    for i in range(h):
        for j in range(w):
            y_min = max(0, i - kernel_r)
            y_max = min(i + kernel_r + 1, h)
            x_min = max(0, j - kernel_r)
            x_max = min(j + kernel_r + 1, w)
            s = (y_max + 1 - y_min) * (x_max + 1 - x_min)
            g[i][j] = np.sum(g_0[y_min:y_max][x_min:x_max]) / s

    while np.any(mask == 0):
        contours, _ = cv.findContours(mask, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_NONE)
        contour = contours[0]
        point_target = [-1, -1]
        priority_max = 0
        for p_c in contour:
            # it is assumed that object is far enough from the edge, otherwise y_min etc precalculation is required
            gamma = np.sum(confidence[p_c[0] - kernel_r: p_c[0] + kernel_r + 1][p_c[1] - kernel_r: p_c[1] + kernel_r + 1]) / area
            priority = gamma * g[p_c[0]][p_c[1]]
            if priority > priority_max:
                priority_max = priority
                point_target = p_c

        point_source = [-1, -1]
        similarity_min = 2000 * area

        for i in range(kernel_r, h - kernel_r):
            for j in range(kernel_r, w - kernel_r):
                # skip if current source patch has empty pixels
                if np.any(confidence[i - kernel_r: i + kernel_r + 1][j - kernel_r: j + kernel_r + 1] == 0):
                    continue

                d_color = 0
                d_grad = 0
                for patch_y in range(-kernel_r, kernel_r + 1):
                    for patch_x in range(-kernel_r, kernel_r + 1):
                        d_color += np.sum(np.abs(np.subtract(img[i + patch_y][j + patch_x], img[point_target[0] + patch_y][point_target[1] + patch_x])))
                        d_grad += abs(g[i + patch_y][j + patch_x] - g[point_target[0] + patch_y][point_target[1] + patch_x])

                # similarity = math.exp(d_color + d_grad)
                similarity = d_color + d_grad
                if similarity < similarity_min:
                    similarity_min = similarity
                    point_source = [i, j]
        break


if __name__ == '__main__':
    img = cv.imread('images/full/car_6895.jpg')
    mask = cv.imread('images/new_masks/car_6895_T640YX180.jpg', cv.IMREAD_GRAYSCALE)
    print(mask)
    print('mask max', mask.max(initial=0))
    inpaint(img, mask)
