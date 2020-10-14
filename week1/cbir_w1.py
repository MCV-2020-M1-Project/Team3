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

    k = 5
    n_bins = 8
    color_space = "RGB"
    distance = "Hellinger"

    print("Computing bbdd histograms...", end=' ', flush=True)
    bbdd_histograms = dbp.compute_bbdd_histograms(bbdd_path, n_bins, color_space)
    print("Done!")
    print('----------------------')

    groundtruth_list = []
    predicted_list = []

    for qsd1_filename in sorted(os.listdir(qsd1_path)):
        if qsd1_filename.endswith('.jpg'):
            image_id = int(qsd1_filename.replace('.jpg', ''))
            k_images = dbp.get_k_images(os.path.join(qsd1_path, qsd1_filename), bbdd_histograms, k, n_bins, distance, color_space)

            print('Image: {}, Groundtruth: {}'.format(qsd1_filename, groundtruth_images[image_id]))
            print('{} most similar images: {}'.format(k, k_images))
            print('----------------------')

            groundtruth_list.append(groundtruth_images[image_id])
            predicted_list.append(k_images)

    print("MAP@{}: {}".format(k, mlm.mapk(groundtruth_list, predicted_list, k)))
    print('*********************************************')
