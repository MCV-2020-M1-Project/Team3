

# cnts = cv.findContours(closed.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#     # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#     cnts = imutils.grab_contours(cnts)
#     cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:4]

#     # loop over the contours from bigger to smaller, and find the biggest one with the right orientation
#     for c in cnts:
#         peri = cv.arcLength(c, True)
#         approx = cv.approxPolyDP(c, 0.015 * peri, True)
#         # if our approximated contour has four points, then
#         # we can assume that we have found our screen
#         # if len(approx) == 4:
#             # cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 10)
#         if image_id == 25:
#             cv.drawContours(img, [approx], -1, (0, 255, 0), 3)
#             cv.imshow(str(randrange(10)),imutils.resize(img, height=600))
#             cv.waitKey()

#         # # # approximate to the rectangle
#         x, y, w, h = cv.boundingRect(c)
#         if w > gray.shape[1]/8 and h > gray.shape[0]/6:
#             # cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 10)
#             # cv.imshow('img',imutils.resize(img, height=600))
#             # cv.waitKey()
#             mask[y:y+h,x:x+w]=255 # fill the mask
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from skimage import feature
import imutils
import os
import operator
import math
# ##----TEST AREA----

def get_angle(box):
    lower_corners=sorted(box, key=operator.itemgetter(1),reverse=True)
    m=(lower_corners[0][1]-lower_corners[1][1])/(lower_corners[0][0]-lower_corners[1][0])
    angle_in_radians = math.atan(m)
    angle_in_degrees = int(-math.degrees(angle_in_radians))
    if(angle_in_degrees<0):
        angle_in_degrees+=180
    width=abs(lower_corners[0][0]-lower_corners[1][0])
    height=abs(lower_corners[0][1]-lower_corners[2][1])
    
    return angle_in_degrees,width,height


def get_mask_M7(img):
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges1 = feature.canny(gray, sigma=3,low_threshold=18,high_threshold=36)
    
    mask=np.zeros(gray.shape)
    mask1=mask.copy()
    mask1[edges1]=255
    mask1=cv.convertScaleAbs(mask1)

    # cv.imshow("mask", imutils.resize(mask1,height=500))
    # cv.imshow("mask2", imutils.resize(mask2,height=500))
    # cv.waitKey()
    
    # 20,20
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(20,20))
    closed = cv.morphologyEx(mask1, cv.MORPH_CLOSE, kernel)

    
    # cv.imshow("closed", imutils.resize(closed,height=500))
    # cv.imshow("closed2", imutils.resize(closed2,height=500))
    # cv.waitKey()
    
    cnts = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:3]
    
    w_img=img.shape[1]
    print(w_img)
    h_img=img.shape[0]
    area_img=w_img*h_img
    
    paintings_coords=[]
    for c in cnts:
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
       
        p1=(box[0][0],box[0][1])
        p2=(box[1][0],box[1][1])
        p3=(box[2][0],box[2][1])
        p4=(box[3][0],box[3][1])
        angle,w_rect,h_rect=get_angle(box)
        
        area_rect=w_rect*h_rect
        
        if(area_rect>0.07*area_img):
            paintings_coords.append([angle,[p1,p2,p3,p4]])
            mask=cv.fillConvexPoly(mask,box,255)
            cv.drawContours(img, [box], 0, (36,255,12), 4)
            
    cv.imshow("img",imutils.resize(img,height=700))
    cv.waitKey()

    return mask, paintings_coords

# query_path = 'data/qsd1_w5/00010.jpg'
# img = cv.imread(query_path)
# mask,coords= get_mask_M7(img)
# print(coords)
query_path = 'data/qsd1_w5'

for query_filename in sorted(os.listdir(query_path)):
    if query_filename.endswith('.jpg'):
        image_id = int(query_filename.replace('.jpg', ''))
        image_path = os.path.join(query_path, query_filename)
        img = cv.imread(image_path)
        mask, coords=get_mask_M7(img)
        print(coords)
        cv.imshow("mask",imutils.resize(mask,height=700))
        cv.waitKey()

