import operator
import pickle
import os

import cv2 as cv
from .dbp import dbp
# import numpy as np

if __name__ == '__main__':

    print('*********************************************')
    print('*********************************************')

    bbdd_path = 'D:\MCV\M1\Project\BBDD'

    qsd1_path = 'D:\MCV\M1\Project\qsd1_w1'

    # load groundtruth images of the query dataset
    groundtruth_images = pickle.load(open(os.path.join(qsd1_path, "gt_corresps.pkl"), 'rb'))


    # for each query image, find the corresponding BBDD image

    bbdd_histograms = dbp.process_images_as_histograms(bbdd_path)

    results = {}
    filename = input("Enter file name of image:")
    file_path = qsd1_path+''
    for qsd1_filename in qsd1_filenames:
        if qsd1_filename.endswith('.jpg'):
            qsd1_number = int(qsd1_filename.replace('.jpg', ''))
            qsd1_hist = dbp.compute_histogram(os.path.join(qsd1_path, qsd1_filename), color_space)
            distances = {}
            for bbdd in bbdd_histograms.keys():
                distances[bbdd] = cv.compareHist(qsd1_hist, bbdd_histograms[bbdd], OPENCV_DISTANCE_METRICS[distance_metric])

            results[qsd1_number] = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:10]

    print(results)