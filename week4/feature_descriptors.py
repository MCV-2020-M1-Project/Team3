import cv2 as cv
import numpy as np
from tqdm import tqdm

import week4.text_boxes as text_boxes

def surf_descriptor(image, threshold=400):
    #Find the SURF keypoints and descriptors of a given image
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create(threshold, )
    kp, des = surf.detectAndCompute(img_gray, None)

    return (kp, des)

def calculate_distance(matches):
    if matches is not None:
        dist = 0
        for m in matches:
            dist += m.distance
        dist /= len(matches)
    else:
        dist = 10000000000000
    return dist

def get_matches_filtered(matches, th=450):
    matches_filtered = []
    for m in matches:
        # print(m.distance)
        if m.distance >= th:
            matches_filtered.append(m)
    return matches_filtered

def orb_descriptor(img,bounding_rm=True):
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    if bounding_rm:
        bounding = text_boxes.detect_text_box(img)
        bounding_mask = np.zeros(img_gray.shape,np.uint8)
        bounding_mask[bounding[1]:bounding[3],bounding[0]:bounding[2]] = 255
        bounding_mask = (255-bounding_mask)
        kp,des=orb.detectAndCompute(img_gray,mask = bounding_mask)
    else:
        kp,des = orb.detectAndCompute(img_gray,mask = None)

    return (kp,des)

def match_descriptors(bbdd_kp_des, query_des, type='BRUTE'):
    # Matching descriptor vectors with a brute force matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    # if type == 'BRUTE':
    #     dm_type = cv.DescriptorMatcher_BRUTEFORCE
    # elif type == 'FLANN':
    #     dm_type = cv.DescriptorMatcher_FLANNBASED
    # #elif type == 'HAMMING':
    # #    dm_type = cv.DescriptorMatcher_BRUTEFORCE_HAMMING
    #
    # matcher = cv.DescriptorMatcher_create(dm_type)
    # matches = matcher.match(d1, d2)

    bd_kp, bd_des = bbdd_kp_des
    if len(bd_kp) > 0:
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(query_des,bd_des)
        # print(len(matches))
        if len(matches) > 150:
            return matches
        else:
            return None
    else:
        return None

def compute_bbdd_orb_descriptors(bbdd_path):
    im = cv.imread(bbdd_path)
    return orb_descriptor(im, False)

def compute_bbdd_orb_query_descriptors(paintings,bbdd_descriptors):
    def calculate_distance(matches):
        dist = 0
        for m in matches:
            dist += m.distance
        return dist/len(matches)

    query_descriptors = []
    query_matches=[]
    for image_id, paintings_image in tqdm(enumerate(paintings), total=len(paintings)):
        for painting_id, painting in enumerate(paintings_image):
            des = orb_descriptor(painting)
            query_matches.append(des)

    return query_matches
