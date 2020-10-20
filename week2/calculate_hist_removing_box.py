import pickle
import os
import ml_metrics as mlm

import histogram as hist

# --------Run area---------
bbdd_path = '../../data/BBDD'
query_path = '../../data/qsd1_w2'


# load groundtruth images and text boxes of the query dataset
groundtruth_images = pickle.load(open(os.path.join(query_path, "gt_corresps.pkl"), 'rb'))
boxes_path=os.path.join(query_path,"text_boxes.pkl")

# parameters: k most similar images, n_bins...
k = 5
n_bins = 8
distance = "Hellinger"
color_space = "RGB"

print("Computing bbdd histograms...", end=' ', flush=True)
bbdd_histograms = hist.compute_bbdd_histograms(bbdd_path, n_bins, color_space)
print("Done!")
print('----------------------')

groundtruth_images_list = []
predicted_images_list = []

for query_filename in sorted(os.listdir(query_path)):
    if query_filename.endswith('.jpg'):
        image_id = int(query_filename.replace('.jpg', ''))
        predicted_images = hist.get_k_images_no_boxes(os.path.join(query_path, query_filename),
                                boxes_path,bbdd_histograms, k, n_bins, distance, color_space)

        print('Image: {}, Groundtruth: {}'.format(query_filename, groundtruth_images[image_id]))
        print('{} most similar images: {}'.format(k, predicted_images))
        print('----------------------')

        groundtruth_images_list.append(groundtruth_images[image_id])
        predicted_images_list.append(predicted_images)

print("MAP@{}: {}".format(k, mlm.mapk(groundtruth_images_list, predicted_images_list, k)))    
