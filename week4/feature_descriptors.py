import cv2 as cv
import numpy as np

def surf_descriptor(image, threshold=400):
    #Find the SURF keypoints and descriptors of a given image
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create(threshold, )
    kp, des = surf.detectAndCompute(img_gray, None)

    return (kp, des)

def match_descriptors(d1, d2):
    # Matching descriptor vectors with a brute force matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
    matches = matcher.match(d1, d2)
    return matches

def draw_matches(img1, img2, threshold=400):
    kp1, des1 = surf_descriptor(img1, threshold)
    kp2, des2 = surf_descriptor(img2, threshold)
    matches = match_descriptors(des1, des2)
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, kp1, img2, kp2, matches, img_matches)
    # -- Show detected matches
    cv.imshow('Matches', img_matches)
    cv.waitKey()



