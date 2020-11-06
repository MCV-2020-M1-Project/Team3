import os
import operator
import cv2 as cv
import numpy as np
import pickle
import ntpath

# import metrics

def sift_corner_detection(image_path):

    img = cv.imread(image_path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite('sift_keypoints.jpg',img)