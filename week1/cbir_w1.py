import os
import pickle
import operator
import cv2 as cv
# import numpy as np
from matplotlib import pyplot as plt

OPENCV_COLOR_SPACES = {
    "RGB": cv.COLOR_BGR2RGB,
    "GRAY": cv.COLOR_BGR2GRAY,
    "LAB": cv.COLOR_BGR2LAB,
    "YCrCb": cv.COLOR_BGR2YCrCb,
    "HSV" : cv.COLOR_BGR2HSV
}

OPENCV_DISTANCE_METRICS = {
    "Correlation": cv.HISTCMP_CORREL,
    "Chi-Squared": cv.HISTCMP_CHISQR,
    "Intersection": cv.HISTCMP_INTERSECT,
    "Hellinger": cv.HISTCMP_BHATTACHARYYA
}

def compute_histogram(image_path, color_space = "RGB"):
    """
    compute_histogram()

    Function to compute ...
    """

    hist_channels = [0]
    hist_bins = [256]
    hist_range = [0,256]

    # if not gray --> 3 channels (RGB, HSV, ...)
    if color_space != "GRAY":
        hist_channels = hist_channels + [1,2]
        hist_bins = hist_bins * 3   # hist_bins = [256, 256, 256]
        hist_range = hist_range * 3 # hist_range = [0, 256, 0, 256, 0, 256]

    img = cv.imread(image_path)
    hist = cv.calcHist([cv.cvtColor(img, OPENCV_COLOR_SPACES[color_space])], hist_channels, None, hist_bins, hist_range)
    hist = cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX).flatten() # change histogram range from [0,256] to [0,1]

    return hist


if __name__ == '__main__':
    """
    ......
    """

    # it would be useful to have an argument parser so the user can select the color_space and distance_metric
    color_space = "RGB"
    distance_metric = "Hellinger"

    print('*********************************************')
    print('Color space: {}, Distance metric: {}'.format(color_space, distance_metric))
    print('*********************************************')

    bbdd_path = '../data/BBDD/'
    bbdd_filenames = sorted(os.listdir(bbdd_path))

    qsd1_path = '../data/qsd1_w1/'
    qsd1_filenames = sorted(os.listdir(qsd1_path))

    # load groundtruth images of the query dataset
    groundtruth_images = pickle.load(open(os.path.join(qsd1_path, "gt_corresps.pkl"), 'rb'))

    # "reverse" handles how sorting the results dictionary will be performed
    reverse = False
    if distance_metric in ("Correlation", "Intersection"):
        reverse = True

    # for each query image, find the corresponding BBDD image
    for idx, qsd1_filename in enumerate(qsd1_filenames):
        if qsd1_filename.endswith('.jpg'):
            print("Query image: {}, Groundtruth image: {}".format(qsd1_filename, groundtruth_images[idx]))

            qsd1_hist = compute_histogram(os.path.join(qsd1_path, qsd1_filename), color_space)
            results = {}

            for bbdd_filename in bbdd_filenames:
                if bbdd_filename.endswith('.jpg'):
                    bbdd_hist = compute_histogram(os.path.join(bbdd_path, bbdd_filename), color_space)
                    distance = cv.compareHist(qsd1_hist, bbdd_hist, OPENCV_DISTANCE_METRICS[distance_metric])
                    results[bbdd_filename] = distance # Todo: just keep the "k" most similar images

            # sort results by distance value
            results = sorted(results.items(), key=operator.itemgetter(1), reverse = reverse)

            print("Most similar images:")

            for idx, (image,d) in enumerate(results):
                print(" Rank #{} --> Image: {}, Distance: {}".format(idx+1, image, d))
                if idx >= 5:
                    break

            print('-----------------------------------------------')
