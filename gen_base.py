import cv2 as cv


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
    new_plate = gen_base(template, 'M140OE_10')
    cv.imshow('sign', new_plate)
    cv.waitKey(0)
    cv.imwrite('images/new_plates/car_6895.jpg', new_plate)