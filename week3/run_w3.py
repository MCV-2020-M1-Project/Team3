import pickle
import os
import ml_metrics as mlm
import cv2 as cv

import imageToText as itt
import histogram as hist
import masks as masks
import evaluation as evaluation
import bg_removal_methods as bg

def run_task2():
    print('---------------------------------------------')
    
    # Path to bbdd and query datasets
    bbdd_path = 'data/BBDD'
    query_path = 'data/qsd1_w3'

    # Parameters
    distance = "Hellinger"
    color_space = "RGB"
    k = 10 # Retrieve k most similar images
    n_bins = 8 # Number of bins per each histogram channel
    block_size = 16 # Block-based histogram
    method_compute_hist = "M1"

    # Path to results
    results_path = os.path.join(query_path, 'results_' + method_compute_hist)

    # If folder data/qsdX_wX/results doesn't exist -> create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    groundtruth_images = pickle.load(open(os.path.join(query_path, "gt_corresps.pkl"), 'rb'))
    groundtruth_text_boxes_path = os.path.join(query_path, 'text_boxes.pkl')
    
    
    print('**********************')
    print('Text based query retrieval. Dataset: {}, Text detection and comparison'.format(query_path))
    print('**********************')
    print("Retrieving bbdd text...", end=' ', flush=True)

    bbdd_texts=itt.get_bbdd_texts(bbdd_path)
    print("Done!")
    print('**********************')

    
    print('**********************')
    print('Color based query retrieval.Dataset: {}, Method to compute histograms: {}, with text detection: {}'.format(query_path, method_compute_hist))
    print('**********************')
    print("Computing bbdd histograms...", end=' ', flush=True)

    bbdd_histograms = hist.compute_bbdd_histograms(bbdd_path, method_compute_hist, n_bins, color_space, block_size)
    print("Done!")
    print('**********************')
    
    text_boxes = []
    groundtruth_images_list = []
    predicted_images_color_list = []
    predicted_images_text_list = []

    for query_filename in sorted(os.listdir(query_path)):
        if query_filename.endswith('.jpg'):
            image_id = int(query_filename.replace('.jpg', ''))
            image_path = os.path.join(query_path, query_filename)
            image = cv.imread(image_path)
            
            ####image=denoise_function(image)
            [tlx, tly, brx, bry] = masks.detect_text_box(image)
            text_boxes.append([tlx, tly, brx, bry])  
            
            predicted_text_list = itt.get_k_images(image, text_boxes[image_id],bbdd_texts,k=10,distance_metric="Levensthein")
            
            # Retrieves the k most similar images ignoring text bounding boxes
            predicted_color_list = hist.get_k_images(image, bbdd_histograms, text_boxes[image_id],
                                                    method_compute_hist, k, n_bins, distance, color_space, block_size)
            
            groundtruth_images_list.append(groundtruth_images[image_id])
            predicted_images_text_list.append(predicted_text_list)
            predicted_images_color_list.append(predicted_color_list)

    print("MAP@{} using text: {}".format(k, mlm.mapk(groundtruth_images_list, predicted_images_text_list, k)))
    print("MAP@{} using color: {}".format(k, mlm.mapk(groundtruth_images_list, predicted_images_color_list, k)))
    
    
    predicted_text_boxes_path = os.path.join(results_path, 'text_boxes.pkl')
    predicted_text_boxes_outfile = open(predicted_text_boxes_path,'wb')
    pickle.dump(text_boxes, predicted_text_boxes_outfile)
    predicted_text_boxes_outfile.close()

    # Text bounding boxes evaluation

    mean_iou = evaluation.mean_iou(query_path, groundtruth_text_boxes_path, predicted_text_boxes_path)
    print('**********************')
    print('Text bounding boxes evaluation: Mean IOU = {}'.format(mean_iou))
    print('**********************')
