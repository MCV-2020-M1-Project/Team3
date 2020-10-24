import os
import imutils
import cv2 as cv
import numpy as np

import week1.bg_removal_methods as methods

def detect_text_box(image):
    # Algorithm to detect bounding box in an image...
    tlx = 0
    tly = 0
    brx = 0
    bry = 0

    return [tlx, tly, brx, bry]

def mask_evaluation(mask_path, groundtruth_path):
    """
    mask_evaluation()

    Function to evaluate a mask at a pixel level...
    """

    mask = cv.imread(mask_path,0) / 255
    mask_vector = mask.reshape(-1)

    groundtruth = cv.imread(groundtruth_path,0) / 255
    groundtruth_vector = groundtruth.reshape(-1)

    tp = np.dot(groundtruth_vector, mask_vector) # true positive
    fp = np.dot(1-groundtruth_vector, mask_vector) # false positive
    tn = np.dot(1-groundtruth_vector, 1-mask_vector) # true negative
    fn = np.dot(groundtruth_vector, 1-mask_vector) # false positive

    epsilon = 1e-10
    precision = tp / (tp+fp+epsilon)
    recall = tp / (tp+fn+epsilon)
    f1 = 2 * (precision*recall) / (precision+recall+epsilon)

    return precision, recall, f1

def mask_average_evaluation(masks_path, groundtruth_path):
    """
    mask_average_evaluation()

    Function to evaluate all masks at a pixel level...
    """

    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    n_masks = 0

    for mask_filename in sorted(os.listdir(masks_path)):
        if mask_filename.endswith('.png'):
            precision, recall, f1 = mask_evaluation(os.path.join(masks_path, mask_filename),
                                                        os.path.join(groundtruth_path, mask_filename.split(".")[0]+'.png'))
            print(" Mask #{} --> Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}".format(mask_filename.split(".")[0], precision, recall, f1))

            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            n_masks += 1

    return avg_precision/n_masks, avg_recall/n_masks, avg_f1/n_masks

def compute_fg(image_path, bg_mask_path, painting_id, bg_results_path):
    """
    compute_fg()

    Function to compute the foreground of an image using the background mask ...
    """

    # Combine the image and the background mask
    combined = cv.bitwise_and(cv.imread(image_path), cv.imread(bg_mask_path))

    gray_combined = cv.cvtColor(combined,cv.COLOR_BGR2GRAY)

    # Coordinates of non-black pixels.
    coords = np.argwhere(gray_combined)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    # Get the contents of the bounding box.
    fg_image = combined[x0:x1, y0:y1]

    # Save foreground image
    cv.imwrite(bg_results_path, fg_image)

    return fg_image

def compute_bg_mask(image_path, bg_mask_path, method="M0", color_space="RGB"):
    """
    compute_bg_mask()

    Function to compute a background mask using an specific method...
    """

    image = cv.imread(image_path)

    if method == "M0":
        mask = methods.get_mask_M0(image)

    elif method == "M1":
        mask = methods.get_mask_M1(image, color_space)

    elif method == "M2":
        mask = methods.get_mask_M2(image, color_space)

    elif method == "M3":
        mask = methods.get_mask_M3(image)

    elif method == "M4":
        mask = methods.get_mask_M4(image)

    # Save background mask
    cv.imwrite(bg_mask_path, mask)

    return 1
