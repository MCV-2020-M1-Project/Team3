import os
import operator
import cv2 as cv
import numpy as np
import pickle
import ntpath

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
        # distances[bbdd_id] = metrics.chi2_distance(qsd1_hist, bbdd_hist)

    k_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]
    return [bbdd_img[0] for bbdd_img in k_images]


#---Week 2 -------
def compute_histogram_no_boxes(image_path, boxes_path, n_bins, color_space="RGB"):
    """
    compute_histogram()

    Function to compute ...
    """

    img = cv.cvtColor(cv.imread(image_path), OPENCV_COLOR_SPACES[color_space])
    image_id = int(ntpath.basename(image_path.replace('.jpg', '')))
    boxes = pickle.load(open(os.path.join(boxes_path), 'rb'))[image_id][0]
    tl=boxes[0] # change to our format results
    br=boxes[2] # change to our format results
        
    n_channels = 1 if color_space == "GRAY" else img.shape[2]
    
    vector_size=img.shape[0]*img.shape[1]-((br[1]-tl[1])*(br[0]-tl[0]))
    result_vector = np.zeros((vector_size,3))
    vector_index = 0
    
    for x in range(img.shape[1]-1):
        if not (tl[1]<x<br[1]):
            for y in range(img.shape[0]-1):
                if not (tl[0]<y<br[0]):
                    result_vector[vector_index,:]=img[y,x,:]
            
    hist_channels = list(range(n_channels))
    hist_bins = [n_bins,]*n_channels
    hist_range = [0, 256]*n_channels

    hist = cv.calcHist([img], hist_channels, None, hist_bins,
                       hist_range)
    hist = cv.normalize(hist, hist, alpha=0, beta=1,
                        norm_type=cv.NORM_MINMAX).flatten()  # change histogram range from [0,256] to [0,1]
    return hist



def get_k_images_no_boxes(qsd1_image_path,boxes_path, bbdd_histograms, k="10", n_bins=8, distance_metric="Hellinger", color_space="RGB"):
    reverse = True if distance_metric in ("Correlation", "Intersection") else False

    qsd1_hist = compute_histogram_no_boxes(qsd1_image_path, boxes_path, n_bins, color_space)
    distances = {}

    for bbdd_id, bbdd_hist in bbdd_histograms.items():
        distances[bbdd_id] = cv.compareHist(qsd1_hist, bbdd_hist, OPENCV_DISTANCE_METRICS[distance_metric])
        # distances[bbdd_id] = metrics.chi2_distance(qsd1_hist, bbdd_hist)

    k_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]
    return [bbdd_img[0] for bbdd_img in k_images]
    
    
    
    
    
    
    