import math

import numpy as np
import cv2 as cv
import pyamg


def extend_mask(mask, ext_radius=3):
    new_mask = mask.copy()
    n = mask.shape[0]
    m = mask.shape[1]
    for i in range(n):
        for j in range(m):
            if not mask[i][j]:
                for ki in range(max(0, i - ext_radius), min(n, i + 2 * ext_radius + 1)):
                    for kj in range(max(0, j - ext_radius), min(m, j + ext_radius + 1)):
                        new_mask[ki][kj] = False
    return new_mask


def inpaint(img, mask):
    kernel_size = 9
    kernel_r = (kernel_size - 1) // 2
    area = kernel_size * kernel_size
    h, w = img.shape[0:2]
    print(h, w)
    # img_8uc1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_8uc1 = img

    confidence: np.ndarray = mask.astype(dtype='int')
    confidence //= confidence.max(initial=0)
    # img[confidence == 0] = [0, 0, 0]
    img[confidence == 0] = 0

    # for proper contour finding, to have the object black, the background white
    # and the contour (contours[0], there are 2 of them) as pixels of the object itself, not outside:
    mask = mask.astype(dtype='int')
    mask -= 255
    mask *= -1
    mask = mask.astype(dtype='uint8')

    gx, gy = cv.spatialGradient(img_8uc1, ksize=3)

    gx = gx.astype(dtype='int64')
    gy = gy.astype(dtype='int64')

    print(gx)
    print(gy)

    # cv.imshow('gx', gx)
    # cv.waitKey(0)
    # cv.imshow('gy', gy)
    # cv.waitKey(0)

    g = np.empty(shape=img.shape[0:2])
    g_0 = np.sqrt(np.add(np.square(gx), np.square(gy)))
    for i in range(h):
        for j in range(w):
            y_min = max(0, i - kernel_r)
            y_max = min(i + kernel_r + 1, h)
            x_min = max(0, j - kernel_r)
            x_max = min(j + kernel_r + 1, w)
            s = (y_max - y_min) * (x_max - x_min)
            g[i][j] = np.sum(g_0[y_min:y_max, x_min:x_max]) / s

    print(g)

    while np.any(confidence == 0):
        print(len(np.where(confidence == 0)[0]), ' pixels to fill')
        print('finding source patch...')
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contour = contours[-1]
        pty, ptx = -1, -1
        priority_max = -1
        print('points in contour: ', len(contour))
        for p_c in contour:
            p_c = p_c[0]
            # it is assumed that object is far enough from the edge, otherwise y_min etc precalculation is required
            gamma = np.sum(confidence[p_c[1] - kernel_r: p_c[1] + kernel_r + 1, p_c[0] - kernel_r: p_c[0] + kernel_r + 1]) / area
            priority = gamma * g[p_c[1]][p_c[0]]
            # print('gamma: ', gamma, 'g[p_c[1]][p_c[0]]: ', g[p_c[1]][p_c[0]])
            if priority > priority_max:
                priority_max = priority
                ptx, pty = p_c

        print(pty, ptx)
        if pty == ptx == -1:
            break
        print(confidence[pty - kernel_r: pty + kernel_r + 1, ptx - kernel_r: ptx + kernel_r + 1])

        psy, psx = -1, -1
        similarity_min = 2000 * area

        for i in range(kernel_r, h - kernel_r, 5):
            for j in range(kernel_r, w - kernel_r, 5):
                # skip if current source patch has empty pixels
                if np.any(confidence[i - kernel_r: i + kernel_r + 1, j - kernel_r: j + kernel_r + 1] == 0):
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
                if confidence[pty + i][ptx + j] == 0:
                    g[pty + i][ptx + j] = g[psy + i][psx + j]
                    gx[pty + i][ptx + j] = gx[psy + i][psx + j]
                    gy[pty + i][ptx + j] = gy[psy + i][psx + j]
                    confidence[pty + i][ptx + j] = 1
                    mask[pty + i][ptx + j] = 0

                    img[pty + i][ptx + j] = img[psy + i][psx + j]

    cv.imshow('inpainted_no_poisson', img)
    cv.waitKey(0)
    cv.imwrite('images/inpainted/no_poisson.jpg', img)

    # TODO: probably add something to make all elements positive
    gx -= gx.min(initial=0)
    gy -= gy.min(initial=0)
    gx = gx.astype(dtype='uint8')
    gy = gy.astype(dtype='uint8')

    gx_2, _ = cv.spatialGradient(gx, ksize=3)
    _, gy_2 = cv.spatialGradient(gy, ksize=3)

    gx_2 -= gx_2.min(initial=0)
    gy_2 -= gy_2.min(initial=0)

    print(gx_2)
    print(gy_2)

    div = np.add(gx_2, gy_2)

    A = pyamg.gallery.poisson(img.shape[0:2], format='csr')
    ml = pyamg.ruge_stuben_solver(A)
    b = div.flatten()
    x = ml.solve(b, tol=1e-10)
    inpainted = np.array(x).reshape(img.shape[0:2])

    cv.imshow('inpainted', inpainted)
    cv.waitKey(0)
    cv.imwrite('images/inpainted/poisson.jpg', inpainted)


if __name__ == '__main__':
    img = cv.imread('images/full/car_6895.jpg', cv.IMREAD_GRAYSCALE)
    mask = cv.imread('images/masks/car_6895.jpg', cv.IMREAD_GRAYSCALE)

    img = cv.resize(src=img, dsize=(0, 0), fx=0.5, fy=0.5)
    mask = cv.resize(src=mask, dsize=(0, 0), fx=0.5, fy=0.5)

    mask[mask <= 100] = 0
    mask[mask > 100] = 255

    h, w = img.shape[0:2]
    print(h, w)

    where = np.where(mask == 0)
    y_min, y_max = where[0].min(), where[0].max()
    x_min, x_max = where[1].min(), where[1].max()

    print(y_min, y_max, x_min, x_max)

    y_diff = y_max - y_min
    x_diff = x_max - x_min

    print(max(0, y_min - y_diff), min(y_max + y_diff, h), max(0, x_min - x_diff), min(x_max + x_diff, w))

    mask = extend_mask(mask)

    cropped_img = img[max(0, y_min - y_diff): min(y_max + y_diff, h), max(0, x_min - x_diff): min(x_max + x_diff, w)]
    cropped_mask = mask[max(0, y_min - y_diff): min(y_max + y_diff, h), max(0, x_min - x_diff): min(x_max + x_diff, w)]

    inpaint(cropped_img, cropped_mask)
