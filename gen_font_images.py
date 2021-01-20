import cv2 as cv

if __name__ == '__main__':
    font = cv.imread('images/font.png', cv.IMREAD_GRAYSCALE)
    _, contours, _ = cv.findContours(font, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    font_string = '1234567890ABCDEHKMOPTXY'
    counter = 0
    for c in contours:
        br = cv.boundingRect(c)
        print(br)
        img = font[br[1]:br[1] + br[3], br[0]:br[0] + br[2]]
        cv.imshow('sym', img)
        cv.waitKey(0)
        sym = input()
        if sym != 'no':
            cv.imwrite(sym + '.png', img)