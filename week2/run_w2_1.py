import pickle
import os
import ml_metrics as mlm
import cv2 as cv

import week1.histogram as hist
import week1.masks as masks
import week1.evaluation as evaluation

def run():
    print('---------------------------------------------')

    # Path to bbdd and query datasets
    bbdd_path = 'data/BBDD'
    query_path = 'data/qsd1_w2'
    results_path = os.path.join(query_path, 'results')

    # If folder data/qsdX_wX/results doesn't exist -> create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Parameters
    distance = "Hellinger"
    color_space = "RGB"
    k = 5 # Retrieve k most similar images
    n_bins = 8 # Number of bins per each histogram channel
    block_size = 16 # Block-based histogram
    method = "M4" # Method to perform background removal

    # Flags to select algorithms
    bg_removal = False
    text_detection = True

    # Test mode
    test = False

    if bg_removal:
        bg_results_path = os.path.join(results_path, method)
        # If folder data/qsdX_wX/results/MX doesn't exist -> create it
        if not os.path.exists(bg_results_path):
            os.makedirs(bg_results_path)

    if text_detection:
        groundtruth_text_boxes_path = os.path.join(query_path, 'text_boxes.pkl')

    if not test:
        # Load groundtruth images of the query dataset
        groundtruth_images = pickle.load(open(os.path.join(query_path, "gt_corresps.pkl"), 'rb'))

    print('**********************')
    print('Dataset: {}, Background removal: {}, Text detection: {}'.format(query_path, bg_removal, text_detection))
    print('**********************')
    print("Computing bbdd histograms...", end=' ', flush=True)

    bbdd_histograms = hist.compute_bbdd_histograms(bbdd_path, n_bins, color_space, block_size)

    print("Done!")
    print('**********************')

    predicted_images_list = []
    groundtruth_images_list = []
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
                num_paintings = masks.compute_bg_mask(image_path, bg_mask_path, method, color_space)

                for idx in range(num_paintings):
                    # Gets the foreground image (painting). We pass the "idx" argument to define which painting we want.
                    # It also saves the foreground image in the results path
                    fg_image = masks.compute_fg(image_path, bg_mask_path, idx, os.path.join(bg_results_path, query_filename))
                    paintings.append(fg_image)

            # If we don't need to remove the background --> painting = image
            else:
                paintings.append(cv.imread(image_path))

            # If needed, to store the text bounding boxes (up to two) of an image
            text_boxes_image = []

            # For each painting
            for painting_id, painting in enumerate(paintings):

                # If we need to detect the text bounding box of the painting
                if text_detection:
                    [tlx, tly, brx, bry] = masks.detect_text_box(painting)
                    # Format of text_boxes_image: [[tlx1, tly1, brx1, bry1], [tlx2, tly2, brx2, bry2]]
                    text_boxes_image.append([tlx, tly, brx, bry])

                    # Retrieves the k most similar images ignoring text bounding boxes
                    predicted_images = hist.get_k_images(painting, bbdd_histograms, text_boxes_image[painting_id],
                                                    k, n_bins, distance, color_space, block_size)

                else:
                    # Retrieves the k most similar images
                    predicted_images = hist.get_k_images(painting, bbdd_histograms, None,
                                                    k, n_bins, distance, color_space, block_size)

                predicted_images_list.append(predicted_images)

            # Format of text_boxes: [[[tlx1, tly1, brx1, bry1], [tlx2, tly2, brx2, bry2]], [[tlx1, tly1, brx1, bry1]] ...]
            text_boxes.append(text_boxes_image)

            if not test:
                print('Image: {}, Groundtruth: {}'.format(query_filename, groundtruth_images[image_id]))
                print('{} most similar images: {}'.format(k, predicted_images))
                print('----------------------')
                groundtruth_images_list.append(groundtruth_images[image_id])

    if not test:
        print("MAP@{}: {}".format(k, mlm.mapk(groundtruth_images_list, predicted_images_list, k)))
        print('**********************')

    predicted_images_path = os.path.join(results_path, 'result.pkl')
    predicted_images_outfile = open(predicted_images_path,'wb')
    pickle.dump(predicted_images_list, predicted_images_outfile)
    predicted_images_outfile.close()

    if text_detection:
        predicted_text_boxes_path = os.path.join(results_path, 'text_boxes.pkl')
        predicted_text_boxes_outfile = open(predicted_text_boxes_path,'wb')
        pickle.dump(text_boxes, predicted_text_boxes_outfile)
        predicted_text_boxes_outfile.close()

        # Text bounding boxes evaluation
        if not test:
            mean_iou = evaluation.mean_iou(groundtruth_text_boxes_path, predicted_text_boxes_path)
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
run()