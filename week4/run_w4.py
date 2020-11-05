from week4 import sift
from week4 import feature_descriptors as fd
import cv2 as cv
def get_corners():
    query_path = 'data/qsd1_w4'

    image_path = query_path + '/00000.jpg'

    sift.sift_corner_detection(image_path)


def run():
    im1 = cv.imread('../qsd1_w4/00000.jpg')
    im2 = cv.imread('../BBDD/bbdd_00106.jpg')
    fd.draw_matches(im1, im2, 5000)