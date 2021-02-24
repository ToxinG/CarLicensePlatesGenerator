import math

import numpy as np
import cv2 as cv
import pyamg


def inpaint(img, mask):
    kernel_size = 9
    kernel_r = (kernel_size - 1) // 2
    area = kernel_size * kernel_size
    h, w = img.shape[0:2]
    # img_8uc1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_8uc1 = img

    # for proper contour finding, to have the object black, the background white
    # and the contour (contours[0], there are 2 of them) as pixels of the object itself, not outside:
    mask = mask.astype(dtype='int')
    mask -= 255
    mask *= -1
    mask = mask.astype(dtype='uint8')

    confidence: np.ndarray = mask.astype(dtype='int')
    confidence //= confidence.max(initial=0)
    # img[confidence == 0] = [0, 0, 0]
    img[confidence == 0] = 0

    gx, gy = cv.spatialGradient(img_8uc1, ksize=3)
    gx = gx.astype(dtype='uint8')
    gy = gy.astype(dtype='uint8')
    gx_2, _ = cv.spatialGradient(gx, ksize=3)
    _, gy_2 = cv.spatialGradient(gy, ksize=3)
    div = np.add(gx_2, gy_2)

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

    while np.any(confidence == 0):
        print('finding source patch...')
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        print(len(contours))
        contour = contours[-1]
        pty, ptx = -1, -1
        priority_max = 0
        for p_c in contour:
            p_c = p_c[0]
            # it is assumed that object is far enough from the edge, otherwise y_min etc precalculation is required
            gamma = np.sum(confidence[p_c[0] - kernel_r: p_c[0] + kernel_r + 1][p_c[1] - kernel_r: p_c[1] + kernel_r + 1]) / area
            priority = gamma * g[p_c[0]][p_c[1]]
            if priority > priority_max:
                priority_max = priority
                pty, ptx = p_c

        psy, psx = -1, -1
        similarity_min = 2000 * area

        for i in range(kernel_r, h - kernel_r):
            print(i)
            for j in range(kernel_r, w - kernel_r):
                # skip if current source patch has empty pixels
                if np.any(confidence[i - kernel_r: i + kernel_r + 1][j - kernel_r: j + kernel_r + 1] == 0):
                    continue

                d_color = 0
                d_grad = 0
                for patch_y in range(-kernel_r, kernel_r + 1):
                    for patch_x in range(-kernel_r, kernel_r + 1):
                        d_color += np.sum(np.abs(np.subtract(img[i + patch_y][j + patch_x], img[pty + patch_y][ptx + patch_x])))
                        d_grad += abs(g[i + patch_y][j + patch_x] - g[pty + patch_y][ptx + patch_x])

                # similarity = math.exp(d_color + d_grad)
                similarity = d_color + d_grad
                if similarity < similarity_min:
                    similarity_min = similarity
                    psy, psx = i, j
        for i in range(-kernel_r, kernel_r + 1):
            for j in range(-kernel_r, kernel_r + 1):
                if confidence[psy + i][psx + j] == 0:
                    g[pty + i][ptx + j] = g[psy + i][psx + j]
                    confidence[psy + i][psx + j] = 1

    A = pyamg.gallery.poisson(img.shape[0:2], format='csr')
    ml = pyamg.ruge_stuben_solver(A)
    b = div.flatten()
    x = ml.solve(b, tol=1e-10)
    inpainted = np.array(x).reshape(img.shape[0:2])

    cv.imshow('inpainted', inpainted)
    cv.waitKey(0)




if __name__ == '__main__':
    img = cv.imread('images/full/car_6895.jpg', cv.IMREAD_GRAYSCALE)
    mask = cv.imread('images/new_masks/car_6895_T640YX180.jpg', cv.IMREAD_GRAYSCALE)

    img = cv.resize(src=img, dsize=(0, 0), fx=0.25, fy=0.25)
    mask = cv.resize(src=mask, dsize=(0, 0), fx=0.25, fy=0.25)

    mask[mask <= 100] = 0
    mask[mask > 100] = 255

    inpaint(img, mask)
