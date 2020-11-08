import os
import operator
import cv2 as cv
import numpy as np
import pickle
import ntpath
import matplotlib.pyplot as plt


from collections import defaultdict

# import metrics

def sift_corner_detection(image_path, db_image_path, threshold=400):

    img = cv.imread(image_path)
    img2 = cv.imread(db_image_path)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(threshold,)
    kp, des = sift.detectAndCompute(gray,None)

    if des is None:
        return False

    gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp2, des2 = sift.detectAndCompute(gray,None)

    if des2 is None:
        return False
    

    # img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imwrite('sift_keypoints.jpg',img)

    bf = cv.BFMatcher()
    print("Match", image_path, db_image_path)
    print("des1:", des)
    print("des2:", des2)
    matches = bf.knnMatch(des,des2, k=2)


    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img = cv.imread(image_path)
    img2 = cv.imread(db_image_path)
    img3 = cv.drawMatchesKnn(img,kp,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('sift_keypoints_match.jpg',img3)

    if len(good)>5:
        return True
    else:
        return False

    
def process_query(query_list, bbdd_list):
    match_dict = defaultdict(list)
    for query_image in query_list:
        for db_image in bbdd_list:
            if sift_corner_detection(query_image, db_image):
                match_dict[query_image].append(db_image)

    return match_dict
