import pickle
import os
import dbp

import metrics

import ml_metrics as mlm

if __name__ == '__main__':

    print('*********************************************')

    bbdd_path = '../data/BBDD'
    qsd1_path = '../data/qsd1_w1'


    # load groundtruth images of the query dataset
    groundtruth_images = pickle.load(open(os.path.join(qsd1_path, "gt_corresps.pkl"), 'rb'))

    # for each query image, find the corresponding BBDD image
    color_space = "RGB"
    bbdd_histograms = dbp.process_db_as_histograms(bbdd_path, color_space)

    results = []
    histograms = []

    for qsd1_filename in image_set:
        if qsd1_filename.endswith('.jpg'):
            qsd1_number = int(qsd1_filename.replace('.jpg', ''))
            histogram = dbp.compute_histogram(os.path.join(qsd1_path, qsd1_filename), color_space)
            histograms.append([qsd1_number, histogram])
            results.append([qsd1_number, dbp.find_similar(histogram, bbdd_histograms, "Hellinger")])


    for t in results:
        print(t)
    print("MAP@k: ")
    print(mlm.mapk(histograms, results, 200))
