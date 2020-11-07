import os
import operator
import cv2 as cv
import numpy as np
import pickle
import ntpath
from matplotlib import pyplot as plt

def sift_corner_detection(image_path, db_image_path):

    img = cv.imread(image_path)
    img2 = cv.imread(db_image_path)

    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)

    gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp2, des2 = sift.detectAndCompute(gray,None)
    

    # img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imwrite('sift_keypoints.jpg',img)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des,des2, k=2)


    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if len(good)>5:
        return True
    else:
        return False

    
