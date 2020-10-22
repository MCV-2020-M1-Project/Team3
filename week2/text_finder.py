from week2 import masks
import numpy as np
import cv2 as cv


def get_mask(image_path):

    img = cv.imread(image_path)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(img_gray, (5, 5), 0)
    img_thr = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 10)
    dil = cv.dilate(img_thr, kernel)

    contours = cv.findContours(dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_thr = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 10)
    dil = cv.dilate(img_thr, kernel)
    contours2=cv.findContours(dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours2 = contours[0] if len(contours) == 2 else contours[1]


    erode = cv.erode(dil, kernel)
    i=0
    for c in contours:
        area = cv.contourArea(c)
        print("tl:", c[0]," br: ", c[-1])
        if area > 10000:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 3)
            # ROI = image[y:y+h, x:x+w]
            # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            # ROI_number += 1
    for c in contours2:
        area = cv.contourArea(c)
        print("tl:", c[0]," br: ", c[-1])
        if area > 10000:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 3)
    #if area > 10000:
    #    x, y, w, h = cv.boundingRect(c)
    #    boxes.append([(x, y), (x + w, y + h)])
    #mask = np.zeros(np.shape(img))
    #cv.rectangle(mask, boxes[0][0], boxes[0][1], (255, 255, 255), -1)
    #cv.rectangle(mask, boxes[1][0], boxes[1][1], (0, 0, 0), -1)
    #cv.imshow(str(i), mask)
    blackhat = cv.morphologyEx(dil, cv.MORPH_BLACKHAT, kernel)
    sharpened = cv.filter2D(dil, -1, kernel)
    cv.imshow('sharpened', img)

    k = cv.waitKey()



def find_edges(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(img_gray, (5, 5), 0)
    cv.Canny(blurred, 0, 50, apertureSize=5)
    dil = cv.dilate(blurred, (5, 5))
    erode = cv.erode(dil, (5, 5))
