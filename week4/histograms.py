import sys
import operator
import cv2 as cv
import numpy as np
from tqdm import tqdm

from skimage import feature
from scipy.fftpack import dct, idct

import multiprocessing.dummy as mp
from functools import partial

import week4.utils as utils

OPENCV_COLOR_SPACES = {
    "RGB": cv.COLOR_BGR2RGB,
    "GRAY": cv.COLOR_BGR2GRAY,
    "LAB": cv.COLOR_BGR2LAB,
    "YCrCb": cv.COLOR_BGR2YCrCb,
    "HSV" : cv.COLOR_BGR2HSV
}

OPENCV_DISTANCE_METRICS = {
    "correlation": cv.HISTCMP_CORREL,
    "chi-squared": cv.HISTCMP_CHISQR,
    "intersection": cv.HISTCMP_INTERSECT,
    "hellinger": cv.HISTCMP_BHATTACHARYYA
}

def rgb_3d_histogram(img):
    """
    compute_color_histogram()

    Function to compute ...
    """

    img = cv.cvtColor(img, OPENCV_COLOR_SPACES["RGB"])
    n_bins=8
    n_channels=3


    hist_channels = list(range(n_channels))
    hist_bins = [n_bins,]*n_channels
    hist_range = [0, 256]*n_channels

    hist = cv.calcHist([img], hist_channels, None, hist_bins,
                       hist_range)

    hist_norm = cv.normalize(hist, hist, alpha=0, beta=1,
                        norm_type=cv.NORM_MINMAX).flatten()  # change histogram range from [0,256] to [0,1]

    return hist_norm

def lbp_histogram(img):

    num_points=8
    radius=2
    gray = cv.cvtColor(img, OPENCV_COLOR_SPACES["GRAY"])

    lbp = feature.local_binary_pattern(gray, num_points, radius, method="uniform")

    hist = cv.calcHist([lbp.astype(np.uint8)], [0], None, [num_points + 2],
                       [0, num_points + 2])

    hist_norm = cv.normalize(hist, hist, alpha=0, beta=1,
                        norm_type=cv.NORM_MINMAX)
    return hist_norm

# Number of blocks: 16, distance: Intersection, norm: 'ortho', n_coefs: 5 --> 0.89 (qsd1_w1)
def dct_histogram(img):

    def dct2(img, norm):
        return dct(dct(img, axis=0, norm=norm), axis=1, norm=norm)

    def idct2(img, norm):
        return idct(idct(img, axis=0, norm=norm), axis=1, norm=norm)

    norm='ortho'
    n_coefs=5

    gray = cv.cvtColor(img, OPENCV_COLOR_SPACES["GRAY"])

    dct_img = dct2(gray, norm)

    dct_zigzag = np.asarray(utils.zigzag(dct_img, n_coefs))

    hist = cv.calcHist([dct_zigzag.astype(np.uint16)], [0], None, [8], [0, max(dct_zigzag)])

    # hist = hist.astype("float32")

    hist_norm = cv.normalize(hist, hist, alpha=0, beta=1,
                             norm_type=cv.NORM_MINMAX)

    return hist_norm

def hog_histogram(img):

    orientations = 8
    pixels_per_cell = (16,16)
    cells_per_block = (1,1)

    gray = cv.cvtColor(img, OPENCV_COLOR_SPACES["GRAY"])

    hog_coefs = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=False, feature_vector=True)
    hog_coefs *= 256

    hist = cv.calcHist([hog_coefs.astype(np.uint8)], [0], None, [8], [0, 256])

    # hist = hist.astype("float32")

    hist_norm = cv.normalize(hist, hist, alpha=0, beta=1,
                             norm_type=cv.NORM_MINMAX)

    return hist_norm

def wavelet_histogram(img):

    wavelet='db1'
    level=3
    n_coefs=7

    gray = cv.cvtColor(img, OPENCV_COLOR_SPACES["GRAY"])

    extracted_coefs = pywt.wavedec2(gray, wavelet=wavelet, level=level)

    first_coef, *level_coefs = extracted_coefs

    wavelet_coefs = []
    wavelet_coefs.append(first_coef)

    for i in range(level):
        (lh, hl, hh) = level_coefs[i]
        wavelet_coefs.append(lh)
        wavelet_coefs.append(hl)
        wavelet_coefs.append(hh)

    wavelet_coefs = wavelet_coefs[:n_coefs]

    hist_concat = None
    for coef in wavelet_coefs:
        max_range = abs(np.amax(coef))+1
        hist = cv.calcHist([coef.astype(np.uint8)], [0], None, [8], [0, max_range])

        hist_norm = cv.normalize(hist, hist, alpha=0, beta=1,
                                 norm_type=cv.NORM_MINMAX)

        if hist_concat is None:
            hist_concat = hist_norm
        else:
            hist_concat = cv.hconcat([hist_concat, hist_norm])

    return hist_concat


def compute_histogram_blocks(img, descriptor, n_blocks, text_box):
    """
    compute_histogram_blocks()

    Function to compute ...
    """

    if text_box is not None:
        tlx_init = text_box[0]
        tly_init = text_box[1]
        brx_init = text_box[2]
        bry_init = text_box[3]

    sizeX = img.shape[1]
    sizeY = img.shape[0]

    hist_concat = None

    for i in range(0,n_blocks):
        for j in range(0, n_blocks):
            # Image block
            img_cell = img[int(i*sizeY/n_blocks):int(i*sizeY/n_blocks) + int(sizeY/n_blocks) ,int(j*sizeX/n_blocks):int(j*sizeX/n_blocks) + int(sizeX/n_blocks)]

            if text_box is None:
                hist = descriptor(img_cell)

            # If there's a text bounding box ignore the pixels inside it
            else:
                tlx = tlx_init-int(j*sizeX/n_blocks)
                tly = tly_init-int(i*sizeY/n_blocks)
                brx = brx_init-int(j*sizeX/n_blocks)
                bry = bry_init-int(i*sizeY/n_blocks)

                img_cell_vector = []

                for x in range(img_cell.shape[1]-1):
                    for y in range(img_cell.shape[0]-1):
                        if not (tlx < x < brx and  tly < y < bry):
                            img_cell_vector.append(img_cell[y,x,:])

                img_cell_vector = np.asarray(img_cell_vector)

                if img_cell_vector.size!=0:
                    img_cell_matrix = np.reshape(img_cell_vector,(img_cell_vector.shape[0],1,-1))
                    hist = descriptor(img_cell_matrix)

            if hist_concat is None:
                hist_concat = hist
            else:
                hist_concat = cv.hconcat([hist_concat, hist])

    return hist_concat

def compute_histogram_multiresolution(img, descriptor):
    multires_blocks = [1, 4, 8]

    hist_concat = None
    for n_blocks in multires_blocks:
        hist = compute_histogram_blocks(img, descriptor)
        if hist_concat is None:
            hist_concat = hist
        else:
            hist_concat = cv.hconcat([hist_concat, hist])

    return hist_concat

HISTOGRAM = {
    "rgb_3d": rgb_3d_histogram,
    "lbp": lbp_histogram,
    "dct": dct_histogram,
    "hog": hog_histogram,
    "wavelet": partial(wavelet_histogram),
    "rgb_3d_blocks": partial(compute_histogram_blocks, descriptor=rgb_3d_histogram, n_blocks=16),
    "rgb_3d_multiresolution": partial(compute_histogram_multiresolution, descriptor=rgb_3d_histogram),
    "lbp_blocks": partial(compute_histogram_blocks, descriptor=lbp_histogram, n_blocks=16),
    "lbp_multiresolution": partial(compute_histogram_multiresolution, descriptor=lbp_histogram),
    "dct_blocks": partial(compute_histogram_blocks, descriptor=dct_histogram, n_blocks=16),
    "dct_multiresolution": partial(compute_histogram_multiresolution, descriptor=dct_histogram),
    "hog_blocks": partial(compute_histogram_blocks, descriptor=hog_histogram, n_blocks=16),
    "hog_multiresolution": partial(compute_histogram_multiresolution, descriptor=hog_histogram),
    "wavelet_blocks": partial(compute_histogram_blocks, descriptor=wavelet_histogram, n_blocks=8),
    "wavelet_multiresolution": partial(compute_histogram_multiresolution, descriptor=wavelet_histogram)
}

def compute_bbdd_histograms(image_path, descriptor):

    return HISTOGRAM[descriptor](cv.imread(image_path), text_box=None)

def compute_distances(paintings, text_boxes, bbdd_histograms, descriptor, metric, weight):

    reverse = True if metric in ("correlation", "intersection") else False

    histograms = []
    all_distances = []

    print('---Computing color query_histograms and distances---')
    for image_id, paintings_image in tqdm(enumerate(paintings), total=len(paintings)):

        text_boxes_image = text_boxes[image_id]
        distances_image = []
        for painting_id, painting in enumerate(paintings_image):
            if len(text_boxes_image) > painting_id:
                text_box = text_boxes_image[painting_id]
            else:
                text_box = None
            hist = HISTOGRAM[descriptor](painting, text_box=text_box)

            distances_painting = []
            for bbdd_hist in bbdd_histograms:
                dist = cv.compareHist(hist, bbdd_hist, OPENCV_DISTANCE_METRICS[metric])
                if reverse:
                    dist = 1/dist
                distances_painting.append(dist)

            min_distance = min(distances_painting)
            max_distance = max(distances_painting)
            distances_painting_norm = [(d-min_distance) / (max_distance-min_distance)
                                        for d in distances_painting]

            # if reverse:
            #     distances_painting_norm = [1-d for d in distances_painting_norm]

            distances_painting_weight = [d * weight for d in distances_painting]
            distances_image.append(distances_painting_weight)

        all_distances.append(distances_image)
    print('-> Done!')

    return all_distances
