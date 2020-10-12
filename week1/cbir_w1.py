import operator
import pickle
import os
# import os.path
import csv
import cv2 as cv
import dbp
import metrics
import numpy as np

if __name__ == '__main__':

    print('*********************************************')
    print('*********************************************')

    bbdd_path = '../data/BBDD'
    qsd1_path = '../data/qsd1_w1'

    # load groundtruth images of the query dataset
    groundtruth_images = pickle.load(open(os.path.join(qsd1_path, "gt_corresps.pkl"), 'rb'))

    # for each query image, find the corresponding BBDD image

    k = 10
    n_bins = 256
    color_space = "RGB"
    distance = "Hellinger"

    bbdd_histograms = dbp.compute_bbdd_histograms(bbdd_path, n_bins, color_space)

    for qsd1_filename in sorted(os.listdir(qsd1_path)):
        if qsd1_filename.endswith('.jpg'):
            image_id = int(qsd1_filename.replace('.jpg', ''))
            k_images = dbp.get_k_images(os.path.join(qsd1_path, qsd1_filename), bbdd_histograms, k, n_bins, distance, color_space)

            print('----------------------')
            print('Image: {}, Groundtruth: {}'.format(qsd1_filename, groundtruth_images[image_id]))
            print(k_images)
