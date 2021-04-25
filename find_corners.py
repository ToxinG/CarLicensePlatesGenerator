import cv2 as cv
import numpy as np
import os

old_h = 85  # 115 mm
old_w = 385  # 520 mm
new_h = 126  # 170 mm
new_w = 217  # 290 mm

corners_old = [[(new_w - old_w) / 2, (new_h - old_h) / 2], [(new_w - old_w) / 2, (new_h + old_h) / 2],
               [(new_w + old_w) / 2, (new_h - old_h) / 2], [(new_w + old_w) / 2, (new_h + old_h) / 2]]

def intersect(line1, line2):
    # takes two lines as [a, b, c] (Ax + By = C) and returns their intersection point

    det = line1[0] * line2[1] - line2[0] * line1[1]
    x = (line2[1] * line1[2] - line1[1] * line2[2]) / det
    y = (line2[0] * line1[2] - line1[0] * line2[2]) / det
    return np.array([int(abs(x)), int(abs(y))])


def find_corners(mask):

    # gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    gray = mask.copy()
    canny_output = cv.Canny(gray, 50, 150, apertureSize=3)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)

    black = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
    black = cv.drawContours(black, hull_list, 0, 255)

    # cv.imshow('drawing', black)
    # cv.waitKey(0)

    corners = cv.goodFeaturesToTrack(black, 4, 0.01, 10)
    corners = [c[0] for c in corners]
    corners.sort(key=lambda x: x[0])
    left = sorted(corners[:2], key=lambda x: x[1])
    right = sorted(corners[2:], key=lambda x: x[1])
    corners = left + right
    # corners = [[c_i[0] - corners[0][0], c_i[1] - corners[0][1]] for c_i in corners]

    for c in corners:
    #     print(c)
        cv.circle(black, tuple(c), 4, 0, -1)

    contours, _ = cv.findContours(black, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    edge_lines = []  # [a, b, c] (Ax + By = C)
    for cnt in contours:
        rows, cols = black.shape[:2]
        [vx, vy, x1, y1] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01).flatten()
        # print(vx, vy, x1, y1)
        x2 = x1 + vx
        y2 = y1 + vy
        a = y2 - y1
        b = x1 - x2
        edge_lines.append([a, b, b * y1 + a * x1])

        # if abs(vx) > 0.6:
        #     lefty = int((-x1 * vy / vx) + y1)
        #     righty = int(((cols - x1) * vy / vx) + y1)
        #     cv.line(black, (0, lefty), (cols, righty), 100, 2)
        # else:
        #     topx = int((-y1 * vx / vy) + x1)
        #     bottomx = int(((rows - y1) * vx / vy) + x1)
        #     cv.line(black, (topx, 0), (bottomx, rows), 100, 2)

    edge_lines.sort(key=lambda x: abs(x[0]))
    corners_true = [intersect(edge_lines[0], edge_lines[2]),
                    intersect(edge_lines[0], edge_lines[3]),
                    intersect(edge_lines[1], edge_lines[2]),
                    intersect(edge_lines[1], edge_lines[3])]

    # for c in corners_true:
    #     print(c)
    #     print(c.shape, type(c))
    #     cv.circle(black, tuple(c), 3, 255, -1)
    #
    # cv.imshow('drawing', black)
    # cv.waitKey(0)

    corners_true.sort(key=lambda x: x[0])
    left = sorted(corners_true[:2], key=lambda x: x[1])
    right = sorted(corners_true[2:], key=lambda x: x[1])
    corners_true = left + right

    return corners_true


if __name__ == '__main__':
    img_dir = 'images/masks'
    for img_name in sorted(os.listdir(img_dir)):
        print(img_name)
        mask = cv.imread(os.path.join(img_dir, img_name))
        new_plate = cv.imread('images/new_plates/car_6895.jpg')
        try:
            corners = find_corners(mask)
            p_t = cv.getPerspectiveTransform(np.array(corners_old, dtype='float32'),
                                             np.array(corners, dtype='float32'))
            warped = cv.warpPerspective(new_plate, p_t, (mask.shape[1], mask.shape[0]), borderMode=cv.BORDER_TRANSPARENT)
            cv.imwrite(os.path.join('images/results', img_name), warped)
        except Exception:
            pass
