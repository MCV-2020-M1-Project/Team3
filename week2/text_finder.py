from week2 import masks
import numpy as np
import cv2 as cv


def get_mask(image_path):

    img = cv.imread(image_path)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    dilate = cv.morphologyEx(blur, cv.MORPH_DILATE, kernel)
    op = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)

    th = cv.morphologyEx(img_gray, cv.MORPH_TOPHAT, kernel)
    dilate = cv.morphologyEx(th, cv.MORPH_DILATE, kernel)
    close = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)

    img_thr = cv.adaptiveThreshold(close, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 10)

    #close = cv.morphologyEx(img_gray, cv.MORPH_CLOSE, kernel)
    #dilate = cv.morphologyEx(close, cv.MORPH_DILATE, kernel)
    #erode = cv.morphologyEx(dilate, cv.MORPH_ERODE, kernel)
    #open = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel)
    #grad = cv.morphologyEx(open, cv.MORPH_GRADIENT, kernel)

    cv.imshow('img', img_thr)
    contours = cv.findContours(img_thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        area = cv.contourArea(c)
        print("tl:", c[0]," br: ", c[-1])
        if area > 10000:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 3)


    cv.imshow('sharpened', img)

    k = cv.waitKey()



def find_edges(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(img_gray, (5, 5), 0)
    cv.Canny(blurred, 0, 50, apertureSize=5)
    dil = cv.dilate(blurred, (5, 5))
    erode = cv.erode(dil, (5, 5))
