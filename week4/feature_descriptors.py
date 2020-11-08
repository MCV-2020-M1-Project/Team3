import cv2 as cv
import numpy as np

def surf_descriptor(image, threshold=400):
    #Find the SURF keypoints and descriptors of a given image
    im = cv.imread(image)
    img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
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



