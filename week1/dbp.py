import os
import operator
import cv2 as cv
import numpy as np

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

def compute_histogram(image_path, n_bins, color_space="RGB"):
    """
    compute_histogram()

    Function to compute ...
    """

    img = cv.imread(image_path)

    n_channels = img.shape[2]
    hist_channels = list(range(n_channels))
    hist_bins = [n_bins,]*n_channels
    hist_range = [0, 256]*n_channels

    hist = cv.calcHist([cv.cvtColor(img, OPENCV_COLOR_SPACES[color_space])], hist_channels, None, hist_bins,
                       hist_range)
    hist = cv.normalize(hist, hist, alpha=0, beta=1,
                        norm_type=cv.NORM_MINMAX).flatten()  # change histogram range from [0,256] to [0,1]
    return hist

def compute_bbdd_histograms(bbdd_path, n_bins=8, color_space="RGB"):
    image_set = sorted(os.listdir(bbdd_path))
    histograms = {}

    for image_filename in image_set:
        if image_filename.endswith('.jpg'):
            image_id = int(image_filename.replace('.jpg', '').replace('bbdd_', ''))
            hist = compute_histogram(os.path.join(bbdd_path, image_filename), n_bins, color_space)
            histograms[image_id] = hist

    return histograms

def get_k_images(qsd1_image_path, bbdd_histograms, k="10", n_bins=8, distance_metric="Hellinger", color_space="RGB"):

    reverse = True if distance_metric in ("Correlation", "Intersection") else False

    qsd1_hist = compute_histogram(qsd1_image_path, n_bins, color_space)
    distances = {}

    for bbdd_id, bbdd_hist in bbdd_histograms.items():
        distances[bbdd_id] = cv.compareHist(qsd1_hist, bbdd_hist, OPENCV_DISTANCE_METRICS[distance_metric])

    k_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]
    return [bbdd_img[0] for bbdd_img in k_images]
