import cv2 as cv
import numpy as np

def surf_descriptor(image, threshold=400):
    #Find the SURF keypoints and descriptors of a given image
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create(threshold, )
    kp, des = surf.detectAndCompute(img_gray, None)

    return (kp, des)

def match_descriptors(d1, d2, type='BRUTE'):
    # Matching descriptor vectors with a brute force matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    if type == 'BRUTE':
        dm_type = cv.DescriptorMatcher_BRUTEFORCE
    elif type == 'FLANN':
        dm_type = cv.DescriptorMatcher_FLANNBASED
    #elif type == 'HAMMING':
    #    dm_type = cv.DescriptorMatcher_BRUTEFORCE_HAMMING

    matcher = cv.DescriptorMatcher_create(dm_type)
    matches = matcher.match(d1, d2)

    return matches

def draw_matches(img1, img2, threshold=400):
    kp1, des1 = surf_descriptor(img1, threshold)
    kp2, des2 = surf_descriptor(img2, threshold)
    img_gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img_gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    matches = match_descriptors(des1, des2)
    img_matches = np.empty((max(img_gray1.shape[0], img_gray2.shape[0]), img_gray1.shape[1] + img_gray2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_gray1, kp1, img_gray2, kp2, matches, img_matches)
    # -- Show detected matches
    cv.imshow('Matches', img_matches)
    cv.waitKey()



