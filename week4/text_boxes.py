import cv2 as cv
import numpy as np
import imutils
import os

import week4.utils as utils

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

    def _get_constrain_rectangles(img):
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
                  # cv.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 5)
                  rectangles_aux.append((x,y,w,h))

        return rectangles_aux

    closed_top_hat=cv.morphologyEx(top_hat,cv.MORPH_CLOSE, kernel)
    closed_black_hat=cv.morphologyEx(black_hat,cv.MORPH_CLOSE, kernel)

    rectangles_top = _get_constrain_rectangles(closed_top_hat)
    rectangles_black = _get_constrain_rectangles(closed_black_hat)

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

    best_boxes_lab.append(get_best_box(l, 60, 30, 150, 120, 30))
    best_boxes_lab.append(get_best_box(a, 60, 30, 30, 120, 30))
    best_boxes_lab.append(get_best_box(b, 60, 30, 20, 120, 30))

    for best_box_lab in best_boxes_lab:
        if best_box_lab is not None:
            [x,y,w,h] = best_box_lab
            # cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 5)

    final_best_box = get_best_rectangle(best_boxes_lab, img)

    # print(f'best_boxes_lab: {best_boxes_lab} --> FINAL: {final_best_box_aux}')

    if final_best_box is not None:

        [x,y,w,h] = final_best_box

        final_best_box = [x, y, x+w, y+h]

        [tlx,tly,brx,bry] = final_best_box
        cv.rectangle(img, (tlx, tly), (brx, bry), (0,255,0), 10)

    # print(final_best_box)

    return final_best_box

# PROBLEM!!!!! TEXT BOXES COORDINATES (SI LES GUARDO AMB EL DESPLAÇAMENT --> HE DE RESTARLO DESPRES O ALGO PER REMOVE TEXT)
def remove_text(paintings, paintings_coords, params, image_id):
    text_boxes = []
    text_boxes_pre = []

    for painting_id in range(len(paintings)):
        painting_path = os.path.join(params['paths']['results'], image_id + '_' + str(painting_id) + '.jpg')
        painting = cv.imread(painting_path)

        text_box = detect_text_box(painting)
        text_boxes_pre.append(text_box)
        if text_box is not None:
            [tlx, tly, brx, bry] = text_box

            # If there is more than one painting, when detecting the text bouning box
            # we have to shift the coordinates so that they make sense in the initial image
            tlx += paintings_coords[painting_id][0]
            tly += paintings_coords[painting_id][1]
            brx += paintings_coords[painting_id][0]
            bry += paintings_coords[painting_id][1]
            text_box = [tlx, tly, brx, bry]
        else:
            text_box = None
        text_boxes.append(text_box)

    utils.save_pickle(os.path.join(params['paths']['results'], 'text_boxes.pkl'), text_boxes)

    return [paintings, text_boxes_pre]
