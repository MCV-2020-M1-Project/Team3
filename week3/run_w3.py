import pickle
import os
import ml_metrics as mlm
import cv2 as cv

import operator

from tqdm import tqdm

import week3.histogram_w3 as hist
import week1.masks as masks
import week1.evaluation as evaluation
import week1.bg_removal_methods as bg

TEXTURE_DESCRIPTORS_DISTANCES = {
    "LBP": "Correlation",
    "DCT": "Hellinger",
    "HOG": "",
    "WAVELET": ""
}

def run():
    print('---------------------------------------------')

    # #---------------Ian-----------------
    # # Initialize parameters (color_ponderation, etc etc) and then create this struct:
    # params = {"color": None, "texture": None, "text": None, "bg_removal": None}
    #
    # if color_retrieval:
    #     params["color"] = {"weight": color_ponderation, "distance_metric": "Hellinger",
    #                        "color_space": "RGB", "n_bins": 8}
    # # We may want to combine different texture descriptors (e.g. LBP + DCT), so we need to add more than one ponderation/descriptor here
    # if texture_retrieval:
    #     params["texture"] = {"weight": texture_ponderation, "descriptor": texture_descriptor,
    #                          "distance_metric": TEXTURE_DESCRIPTORS_DISTANCES[texture_descriptor]}
    #
    # if text_retrieval:
    #     params["text"] = {"weight": text_ponderation, "distance_metric": "????"}
    #
    # if bg_removal:
    #     params["bg_removal"] = {"method": "M5"}
    #
    # # Then in compute_bbdd_histograms function (for example) we can only pass the params struct
    # # and check if a descriptor is needed (e.g. if params["texture"] is not None) and then access to the param
    # # values like params["texture"]["descriptor"]
    # #---------------Ian-----------------

    # Path to bbdd and query datasets
    bbdd_path = 'data/BBDD'
    query_path = 'data/qsd1_w1'

    # Flags to select algorithms
    bg_removal = False
    text_detection = False

    # Test mode
    test = False

    # Parameters
    distance_color = "Hellinger"
    distance_texture_DCT = "Intersection"
    distance_texture_WAV = "Hellinger"
    distance_texture_LBP = "Correlation"
    distance_texture_HOG = "Hellinger"

    block_size_color = 16
    block_size_texture_DCT = 16
    block_size_texture_WAV = 8
    block_size_texture_LBP = 16
    block_size_texture_HOG = 16

    color_space = "RGB"
    k = 1 # Retrieve k most similar images
    n_bins = 8 # Number of bins per each histogram channel
    method_compute_hist = "M1"
    method_bg = "M5" # Method to perform background removal

    # Path to results
    results_path = os.path.join(query_path, 'results_' + method_compute_hist)

    # If folder data/qsdX_wX/results doesn't exist -> create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if bg_removal:
        bg_results_path = os.path.join(results_path, 'bg_removal_' + method_bg)
        # If folder data/qsdX_wX/results/MX doesn't exist -> create it
        if not os.path.exists(bg_results_path):
            os.makedirs(bg_results_path)

    if text_detection:
        groundtruth_text_boxes_path = os.path.join(query_path, 'text_boxes.pkl')

    if not test:
        # Load groundtruth images of the query dataset
        groundtruth_paintings = pickle.load(open(os.path.join(query_path, "gt_corresps.pkl"), 'rb'))

    print('**********************')
    print('Dataset: {}, Method to compute histograms: {}, Background removal: {}, Text detection: {}'.format(query_path, method_compute_hist, bg_removal, text_detection))
    print('**********************')
    print("Computing bbdd histograms...")

    # bbdd_histograms_color = hist.compute_bbdd_histograms_color(bbdd_path, method_compute_hist, n_bins, color_space, block_size_color)

    # bbdd_histograms_texture_DCT = hist.compute_bbdd_histograms_texture(bbdd_path, method_compute_hist, "DCT", n_bins, color_space, block_size_texture_DCT)
    # bbdd_histograms_texture_WAV = hist.compute_bbdd_histograms_texture(bbdd_path, method_compute_hist, "WAVELET", n_bins, color_space, block_size_texture_WAV)
    # bbdd_histograms_texture_LBP = hist.compute_bbdd_histograms_texture(bbdd_path, method_compute_hist, "LBP", n_bins, color_space, block_size_texture_LBP)
    bbdd_histograms_texture_HOG = hist.compute_bbdd_histograms_texture(bbdd_path, method_compute_hist, "HOG", n_bins, color_space, block_size_texture_HOG)

    # bbdd_texture_histograms = hist.compute_bbdd_histograms_texture(bbdd_path, method_compute_hist, n_bins, color_space, block_size)

    print("Done!")
    print('**********************')

    predicted_paintings_list = []
    groundtruth_paintings_list = []
    text_boxes = []

    # For each image of the query dataset, we remove the background (if needed), detect
    # the text bounding boxes (if needed), and compare the painting (or paintings, if there are two)
    # to each image of the bbdd dataset to retrieve the k most similar images
    for query_filename in sorted(os.listdir(query_path)):
        if query_filename.endswith('.jpg'):
            image_id = int(query_filename.replace('.jpg', ''))
            image_path = os.path.join(query_path, query_filename)

            # If needed, to store the painting/s (up to two) of an image
            paintings = []

            # If we need to remove the background
            if bg_removal:
                bg_mask_path = os.path.join(bg_results_path, query_filename.split(".")[0]+'.png')

                # Computes the background mask and returns the number of paintings in the image.
                # It also saves the background mask in the results path
                paintings_coords = masks.compute_bg_mask(image_path, bg_mask_path, method_bg, color_space)

                for painting_coords in paintings_coords:
                    # Gets the foreground image (painting). We pass the "idx" argument to define which painting we want.
                    # It also saves the foreground image in the results path
                    painting = masks.compute_fg(image_path, bg_mask_path, painting_coords, os.path.join(bg_results_path, query_filename))
                    paintings.append(painting)

            # If we don't need to remove the background --> painting = image
            else:
                paintings.append(cv.imread(image_path))

            # If needed, to store the text bounding boxes (up to two) of an image
            text_boxes_image = []

            # To store the paintings (up to two) k similar images
            predicted_paintings_per_image = []

            # For each painting
            for painting_id, painting in enumerate(paintings):

                # If we need to detect the text bounding box of the painting
                if text_detection:
                    [tlx, tly, brx, bry], _ = masks.detect_text_box(painting)

                    # If there are two paintings, when detecting the text bouning box of the
                    # second one we have to shift the coordinates so that they make sense in the initial image
                    if bg_removal:
                        tlx += paintings_coords[painting_id][0]
                        tly += paintings_coords[painting_id][1]
                        brx += paintings_coords[painting_id][0]
                        bry += paintings_coords[painting_id][1]

                    text_boxes_image.append([tlx, tly, brx, bry])

                    # Retrieves the k most similar images ignoring text bounding boxes
                    # _, distances_color = hist.get_k_images_color(painting, bbdd_histograms_color, text_boxes_image[painting_id],
                    #                                 method_compute_hist, k, n_bins, distance_color, color_space, block_size_color)

                    # _, distances_texture_DCT = hist.get_k_images_texture(painting, "DCT", bbdd_histograms_texture_DCT, text_boxes_image[painting_id],
                    #                                 method_compute_hist, k, n_bins, distance_texture_DCT, color_space, block_size_texture_DCT)

                    # _, distances_texture_WAV = hist.get_k_images_texture(painting, "WAVELET", bbdd_histograms_texture_WAV, text_boxes_image[painting_id],
                    #                                 method_compute_hist, k, n_bins, distance_texture_WAV, color_space, block_size_texture_WAV)

                    # _, distances_texture_LBP = hist.get_k_images_texture(painting, "LBP", bbdd_histograms_texture_LBP, text_boxes_image[painting_id],
                    #                                 method_compute_hist, k, n_bins, distance_texture_LBP, color_space, block_size_texture_LBP)

                    _, distances_texture_HOG = hist.get_k_images_texture(painting, "HOG", bbdd_histograms_texture_HOG, text_boxes_image[painting_id],
                                                    method_compute_hist, k, n_bins, distance_texture_HOG, color_space, block_size_texture_HOG)

                else:
                    # Retrieves the k most similar images
                    # _, distances_color = hist.get_k_images_color(painting, bbdd_histograms_color, None,
                    #                                 method_compute_hist, k, n_bins, distance_color, color_space, block_size_color)

                    # _, distances_texture_DCT = hist.get_k_images_texture(painting, "DCT", bbdd_histograms_texture_DCT, None,
                    #                                 method_compute_hist, k, n_bins, distance_texture_DCT, color_space, block_size_texture_DCT)

                    # _, distances_texture_WAV = hist.get_k_images_texture(painting, "WAVELET", bbdd_histograms_texture_WAV, None,
                    #                                 method_compute_hist, k, n_bins, distance_texture_WAV, color_space, block_size_texture_WAV)

                    # _, distances_texture_LBP = hist.get_k_images_texture(painting, "LBP", bbdd_histograms_texture_LBP, None,
                    #                                 method_compute_hist, k, n_bins, distance_texture_LBP, color_space, block_size_texture_LBP)

                    _, distances_texture_HOG = hist.get_k_images_texture(painting, "HOG", bbdd_histograms_texture_HOG, None,
                                                    method_compute_hist, k, n_bins, distance_texture_HOG, color_space, block_size_texture_HOG)


                reverse = True if distance_texture_HOG in ("Correlation", "Intersection") else False

                # reverse = False

                color_weight = 0.0
                texture_weight_DCT = 0.0
                texture_weight_WAV = 1.0
                # texture_weight_LBP = 0.2
                # texture_weight_HOG = 0.2
                text_weight = 0.0

                weighted_distances={}
                for key in distances_texture_HOG:
                    # weighted_distances[key]=color_weight*distances_color[key]+texture_weight_DCT*1/(distances_texture_DCT[key]+1e-7)+texture_weight_WAV*distances_texture_WAV[key]+texture_weight_LBP*1/(distances_texture_LBP[key]+1e-7)+texture_weight_HOG*distances_texture_HOG[key]
                    # weighted_distances[key]=color_weight*distances_color[key]+texture_weight_DCT*1/(distances_texture_DCT[key]+1e-7)+texture_weight_WAV*distances_texture_WAV[key]
                    # weighted_distances[key]=color_weight*distances_color[key]+texture_weight_DCT*1/(distances_texture_DCT[key]+1e-7)
                    weighted_distances[key]=distances_texture_HOG[key]

                k_predicted_images = (sorted(weighted_distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]

                predicted_imgs_aux = [predicted_image[0] for predicted_image in k_predicted_images]

                predicted_paintings_per_image.append(predicted_imgs_aux)

            if not test:
                print('Image: {}'.format(query_filename))
                for painting_id, groundtruth_painting in enumerate(groundtruth_paintings[image_id]):
                    print('-> Painting #{}'.format(painting_id))
                    print('    Groundtruth: {}'.format(groundtruth_painting))

                    # If we detected the painting
                    if len(predicted_paintings_per_image) > painting_id:
                        print('        {} most similar images: {}'.format(k, predicted_paintings_per_image[painting_id]))
                    else:
                        print('        Painting not detected!!')
                        predicted_paintings_per_image.append([0,0,0,0,0])

                print('----------------------')
                groundtruth_paintings_list.append(groundtruth_paintings[image_id])

            predicted_paintings_list.append(predicted_paintings_per_image)

            # Format of text_boxes: [[[tlx1, tly1, brx1, bry1], [tlx2, tly2, brx2, bry2]], [[tlx1, tly1, brx1, bry1]] ...]
            text_boxes.append(text_boxes_image)


    if not test:

        # Adapt the format of the "gt_corresps.pkl" to be able to evaluate the results...
        groundtruth_paintings_list_eval = []
        predicted_paintings_list_eval = []

        if 'qsd2_w2' in query_path or 'qsd2_w3' in query_path:
            for groundtruth_paintings_per_image in groundtruth_paintings_list:
                for groundtruth_painting in groundtruth_paintings_per_image:
                    groundtruth_paintings_list_eval.append([groundtruth_painting])
        else:
            groundtruth_paintings_list_eval = groundtruth_paintings_list

        for predicted_paintings_per_image in predicted_paintings_list:
            for predicted_paintings_per_painting in predicted_paintings_per_image:
                predicted_paintings_list_eval.append(predicted_paintings_per_painting)

        print("MAP@{}: {}".format(k, mlm.mapk(groundtruth_paintings_list_eval, predicted_paintings_list_eval, k)))
        print('**********************')

    predicted_paintings_path = os.path.join(results_path, 'result.pkl')
    predicted_paintings_outfile = open(predicted_paintings_path,'wb')
    pickle.dump(predicted_paintings_list, predicted_paintings_outfile)
    predicted_paintings_outfile.close()

    if text_detection:
        predicted_text_boxes_path = os.path.join(results_path, 'text_boxes.pkl')
        predicted_text_boxes_outfile = open(predicted_text_boxes_path,'wb')
        pickle.dump(text_boxes, predicted_text_boxes_outfile)
        predicted_text_boxes_outfile.close()

        # Text bounding boxes evaluation
        if not test:
            mean_iou = evaluation.mean_iou(query_path, groundtruth_text_boxes_path, predicted_text_boxes_path)
            print('**********************')
            print('Text bounding boxes evaluation: Mean IOU = {}'.format(mean_iou))
            print('**********************')

    # Background removal evaluation
    if bg_removal and not test:
        # The background masks and corresponding foregrounds are saved in 'data/qsdX_wX/results/MX'
        avg_precision, avg_recall, avg_f1 = masks.mask_average_evaluation(bg_results_path, query_path)
        print('**********************')
        print('Average --> Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(avg_precision, avg_recall, avg_f1))
        print('**********************')
    print('---------------------------------------------')
