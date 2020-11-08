import cv2 as cv
import numpy as np
from week4 import text_boxes as tb
from tqdm import tqdm



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

def compute_bbdd_orb_descriptors(bbdd_list):
    bbdd_descriptors = []
    for filename in bbdd_list:
        im = cv.imread(filename)
        ds = orb_descriptor(im, False)
        bbdd_descriptors.append(ds)
    return bbdd_descriptors

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