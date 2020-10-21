import os
import operator
import cv2 as cv
import numpy as np

# import metrics

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

    img = cv.cvtColor(cv.imread(image_path), OPENCV_COLOR_SPACES[color_space])

    n_channels = 1 if color_space == "GRAY" else img.shape[2]

    hist_channels = list(range(n_channels))
    hist_bins = [n_bins,]*n_channels
    hist_range = [0, 256]*n_channels

    hist = cv.calcHist([img], hist_channels, None, hist_bins,
                       hist_range)
    hist = cv.normalize(hist, hist, alpha=0, beta=1,
                        norm_type=cv.NORM_MINMAX).flatten()  # change histogram range from [0,256] to [0,1]
    return hist

def compute_histogram_blocks(image_path, n_bins, color_space="RGB", rows=1,cols=1):
    """
    compute_histogram()

    Function to compute ...
    """

    img = cv.cvtColor(cv.imread(image_path), OPENCV_COLOR_SPACES[color_space])

    sizeX = img.shape[1]
    sizeY = img.shape[0]

    hist_row = None
    hist_concat = None

    for i in range(0,rows):
        for j in range(0, cols):
            # Image block
            img_cell = img[int(i*sizeY/rows):int(i*sizeY/rows) + int(sizeY/rows) ,int(j*sizeX/cols):int(j*sizeX/cols) + int(sizeX/cols)]

            n_channels = 1 if color_space == "GRAY" else img.shape[2]

            hist_channels = list(range(n_channels))
            hist_bins = [n_bins,]*n_channels
            hist_range = [0, 256]*n_channels

            hist = cv.calcHist([img_cell], hist_channels, None, hist_bins,
                            hist_range)
            hist = cv.normalize(hist, hist, alpha=0, beta=1,
                                norm_type=cv.NORM_MINMAX).flatten()  # change histogram range from [0,256] to [0,1]

            if hist_row is None:
                hist_row = hist
            else:
                hist_row = cv.hconcat([hist_row, hist])
        
        if hist_concat is None:
            hist_concat = hist_row
        else:
            hist_concat = cv.vconcat([hist_concat, hist_row])
        
        hist_row = None

    
    return hist_concat

def compute_bbdd_histograms(bbdd_path, n_bins=8, color_space="RGB", rows=1, cols=1):
    image_set = sorted(os.listdir(bbdd_path))
    histograms = {}

    for image_filename in image_set:
        if image_filename.endswith('.jpg'):
            image_id = int(image_filename.replace('.jpg', '').replace('bbdd_', ''))
            hist = compute_histogram_blocks(os.path.join(bbdd_path, image_filename), n_bins, color_space, rows=rows, cols=cols)
            histograms[image_id] = hist

    return histograms

def get_k_images(qsd1_image_path, bbdd_histograms, k="10", n_bins=8, distance_metric="Hellinger", color_space="RGB", rows=1, cols=1):

    reverse = True if distance_metric in ("Correlation", "Intersection") else False

    qsd1_hist = compute_histogram_blocks(qsd1_image_path, n_bins, color_space, rows=rows, cols=cols)
    distances = {}

    for bbdd_id, bbdd_hist in bbdd_histograms.items():
        distances[bbdd_id] = cv.compareHist(qsd1_hist, bbdd_hist, OPENCV_DISTANCE_METRICS[distance_metric])
        # distances[bbdd_id] = metrics.chi2_distance(qsd1_hist, bbdd_hist)

    k_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]
    return [bbdd_img[0] for bbdd_img in k_images]
