import cv2 as cv
import numpy as np
from week4 import text_boxes as tb


def surf_descriptor(image, threshold=400):
    #Find the SURF keypoints and descriptors of a given image
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create(threshold, )
    kp, des = surf.detectAndCompute(img_gray, None)

    return (kp, des)

def orb_descriptor(img,bounding_rm=True):
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    if bounding_rm:
        bounding = tb.detect_text_box(img)
        bounding_mask = np.zeros(img_gray.shape,np.uint8)
        bounding_mask[bounding[1]:bounding[3],bounding[0]:bounding[2]] = 255
        bounding_mask = (255-bounding_mask)
        kp,des=orb.detectAndCompute(img_gray,mask = bounding_mask)
    else:
        kp,des = orb.detectAndCompute(img_gray,mask = None)

    return (kp,des)


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
