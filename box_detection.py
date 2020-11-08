# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import cv2 as cv
import numpy as np
import imutils
import os

def compute_score(rect, img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_height_center = img_height / 2.0
    img_width_center = img_width / 2.0
    aspect_ratio = 4.0

    rect_x = rect[0]
    rect_y = rect[1]
    rect_width = rect[2]
    rect_height = rect[3]
    rect_width_center = rect_x + rect_width/2.0
    rect_height_center = rect_y + rect_height/2.0
    rect_ratio = float(rect_width)/rect_height

    x_center_score = abs(rect_width_center-img_width_center)**2
    if rect_height_center <= img_height_center:
        y_center_score = abs(rect_height_center-img_height/4.0)**2
    else:
        y_center_score = abs(rect_height_center-img_height*3.0/4.0)**2

    ratio_score = abs(rect_ratio-aspect_ratio)**2

    # symmetry_score = abs((rect_x - img_width_center) - ((rect_x + rect_width) - img_width_center))**2

    symmetry_score = abs((rect_x - img_width/5.0)**2 - ((rect_x + rect_width) - img_width*4.0/5.0)**2)**2
    symmetry_center_score = abs((rect_x - img_width_center)**2 - ((rect_x + rect_width) - img_width_center)**2)**2

    x_center_weight = 0.5
    y_center_weight = 0.3
    ratio_weight = 0.25
    symmetry_weight = 0.3
    symmetry_center_weight = 0.1

    distance = x_center_score*x_center_weight + y_center_score*y_center_weight \
               + ratio_score*ratio_weight + symmetry_score*symmetry_weight + symmetry_center_score*symmetry_center_weight

    # print('------------------')
    # print(f'x_c_s: {x_center_score}, y_c_s: {y_center_score}, r_s: {ratio_score}')
    # print(f'symm1: {symmetry_score}, symm2: {symmetry_center_score}')
    # print(f'distance: {distance}')
    # print('------------------')

    return distance

def get_best_rectangle(rectangles, img):
    distances = []
    for rect in rectangles:
        if rect is not None:
            distances.append(compute_score(rect, img))
        else:
            distances.append(100000000000000)

    idx_best_rectangle = np.argsort(distances)[0]
    return rectangles[idx_best_rectangle]

def get_best_box(img, filter1_size_x, filter1_size_y, threshold, filter2_size_x, filter2_size_y):

    # Getting the kernel to be used in Gradient
    filterSize = (filter1_size_x,filter1_size_y)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
    top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

    TH = threshold
    top_hat[(top_hat[:,:] < TH) ] = 0
    black_hat[(black_hat[:,:] < TH) ] = 0

    filterSize = (filter2_size_x,filter2_size_y)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)

    def _get_constrain_rectangles(img, imgshow, aux):
        contours = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        w_img = img.shape[1]
        h_img = img.shape[0]

        rectangles_aux = []

        for c in contours:
              # # approximate to the rectangle
              x, y, w, h = cv.boundingRect(c)
              r = float(cv.countNonZero(img[y:y+h, x:x+w])) / (w * h)
              if (w_img/10 < w < w_img*0.95) and (h_img/40 < h < h_img/2) and (w > h*2) and r > 0.35:
                  cv.rectangle(imgshow, (x, y), (x + w, y + h), (0,0,255), 10)
                  rectangles_aux.append((x,y,w,h))

        # cv.imshow(str(aux), imutils.resize(imgshow, height=600))

        return rectangles_aux

    closed_top_hat=cv.morphologyEx(top_hat,cv.MORPH_CLOSE, kernel)
    closed_black_hat=cv.morphologyEx(black_hat,cv.MORPH_CLOSE, kernel)

    # cv.imshow('top', top_hat)
    # cv.imshow('black', black_hat)
    # cv.imshow('closed_top', closed_top_hat)
    # cv.imshow('closed_black', closed_black_hat)

    rectangles_top = _get_constrain_rectangles(closed_top_hat, img, 0)
    rectangles_black = _get_constrain_rectangles(closed_black_hat, img, 1)
    # cv.waitKey()

    rectangles = rectangles_top + rectangles_black

    if len(rectangles) == 0:
        return None
    else:
        final_rect = get_best_rectangle(rectangles, img)

    return final_rect

def detect_text_box(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l,a,b = cv.split(lab)

    best_boxes_lab = []

    best_boxes_lab.append(get_best_box(l, 90, 60, 150, 120, 30))
    best_boxes_lab.append(get_best_box(a, 60, 30, 30, 120, 30))
    best_boxes_lab.append(get_best_box(b, 60, 30, 20, 120, 30))

    for best_box_lab in best_boxes_lab:
        if best_box_lab is not None:
            [x,y,w,h] = best_box_lab
            cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 5)

    final_best_box = get_best_rectangle(best_boxes_lab, img)

    # print(f'best_boxes_lab: {best_boxes_lab} --> FINAL: {final_best_box_aux}')

    if final_best_box is not None:

        [x,y,w,h] = final_best_box

        final_best_box = [x, y, x+w, y+h]

        [tlx,tly,brx,bry] = final_best_box
        cv.rectangle(img, (tlx, tly), (brx, bry), (0,255,0), 10)
        print(final_best_box)

    # print(final_best_box)

    return final_best_box

query_path = 'data/qsd1_w4_denoised'

for query_filename in sorted(os.listdir(query_path)):
    image_id = int(query_filename.replace('.jpg', ''))
    image_path = os.path.join(query_path, query_filename)
    img = cv.imread(image_path)
    print(image_id)
    # if image_id == 5:
    text_box = detect_text_box(img)
    cv.imshow(str(image_id),imutils.resize(img,height=600))
    cv.waitKey()
