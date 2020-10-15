import os
import imutils
import cv2 as cv
import numpy as np

def get_mask_M1(image_path, color_space="RGB"):
    """
    get_mask_M1()

    Function to compute a binary mask of the background using method 1...
    """

    # load the input image
    image = cv.imread(image_path)

    # Converting color space
    if color_space == "RGB":
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        lower_thr = np.array([0, 0, 0]) # Rmin, Gmin, Bmin
        upper_thr = np.array([100, 100, 100]) # Rmax, Gmax, Bmax

    elif color_space == "HSV":
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower_thr = np.array([0, 100, 0]) # Hmin, Smin, Vmin
        upper_thr = np.array([180, 255, 255]) # Hmax, Smax, Vmax

    mask = cv.inRange(image, lower_thr, upper_thr)

    return mask

def get_mask_M2(image_path, color_space="RGB"):
    """
    get_mask_M2()

    Function to compute a binary mask of the background using method 2...
    """

    mask = get_mask_M1(image_path, color_space)

    # Filling the holes with closing
    dilatation_size = 6
    element = cv.getStructuringElement(cv.MORPH_RECT, (2*dilatation_size+1, 2*dilatation_size+1),
                                        (int(dilatation_size/2), int(dilatation_size/2)))
    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, element, iterations=9)

    # # Filling the holes with imfill flood filling
    # mask_flood = mask.copy()
    # h, w = mask_flood.shape[:2]
    # fill_mask = np.zeros((h+2,w+2), np.uint8)
    #
    # # Filling from the central seed. Might try different seeds
    # cv.floodFill(mask_flood,fill_mask,(h//2,w//2),255)

    return mask_closed

def get_mask_M3(image_path):
    """
    get_mask_M3()

    Function to compute a binary mask of the background using method 3...
    """

    # Tunning parameters. We can put this as input to the function as well
    CANNY_THRESH_1 = 30
    CANNY_THRESH_2 = 130

    # load the input image
    image = cv.imread(image_path,0)
    blurred = cv.GaussianBlur(image, (5, 5), 0)

    # obtain the edges of the image
    edges = cv.Canny(blurred, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv.dilate(edges, None)
    edges = cv.erode(edges, None)

    # find contours in the edged image
    cnts = cv.findContours(edges.copy(), cv.RETR_LIST,
        cv.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    # sort from biggest area to smallest and take the top5
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]


    mask = np.zeros(edges.shape)
    cmax, max_extent=[],0
    # loop over the contours from bigger to smaller, and find the biggest one with the right orientation
    for c in cnts:
          # # approximate to the hull.
          hull = cv.convexHull(c)

          # find the contour with the highest extent compared to the bounding rectangle
          area = cv.contourArea(hull)
          x,y,w,h = cv.boundingRect(c)
          rect_area = w*h
          extent = float(area)/rect_area

          # get the contour with max extent (area covered, approximation area)
          if max_extent<extent:
              max_extent=extent
              cmax=hull

    cv.fillConvexPoly(mask, cmax, (255)) # fill the mask

    return mask

def compute_masks(images_path, method="M1", color_space="RGB"):
    """
    compute_masks()

    Function to compute all masks using an specific method...
    """

    # for each query image, find the corresponding mask
    for image_filename in sorted(os.listdir(images_path)):
        if image_filename.endswith('.jpg'):
            if method == "M1":
                mask = get_mask_M1(os.path.join(images_path, image_filename), color_space=color_space)

            elif method == "M2":
                mask = get_mask_M2(os.path.join(images_path, image_filename), color_space=color_space)

            elif method == "M3":
                mask = get_mask_M3(os.path.join(images_path, image_filename))

            cv.imwrite(os.path.join(images_path+'/results_'+method+'/'+image_filename.split(".")[0]+ '.png'),mask)

def mask_evaluation(mask_path,groundtruth_path):
    """
    mask_evaluation()

    Function to evaluate a mask at a pixel level...
    """

    mask = cv.imread(mask_path,0)/255
    mask_vector = mask.reshape(-1)

    groundtruth = cv.imread(groundtruth_path,0)/255
    groundtruth_vector = groundtruth.reshape(-1)

    TP = np.dot(groundtruth_vector, mask_vector) # true positive
    FP = np.dot(1-groundtruth_vector, mask_vector) # false positive
    TN = np.dot(1-groundtruth_vector, 1-mask_vector) # true negative
    FN = np.dot(groundtruth_vector, 1-mask_vector) # false positive

    epsilon = 1e-10
    precision = TP / (TP+FP+epsilon)
    recall = TP / (TP+FN+epsilon)
    f1 = 2 * (precision*recall) / (precision+recall+epsilon)

    return precision, recall, f1

def mask_average_evaluation(masks_path,groundtruth_path, method="M1"):
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

def get_foreground(image_path,mask_path):
    """
    get_foreground()

    Function to retrieve the masked image without background...
    """
    # load the input and mask images
    image = cv.imread(image_path)
    mask = cv.imread(mask_path)
    
    # combine the mask and the image
    masked = cv.bitwise_and(mask,image)

    # convert the masked image to grayscale
    gray = cv.cvtColor(masked,cv.COLOR_BGR2GRAY)
    
    # Coordinates of non-black pixels.
    coords = np.argwhere(gray)
    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    
    # Get the contents of the bounding box.
    foreground = masked[x0:x1, y0:y1]
    
    return foreground

def compute_foregrounds(images_path,masks_path,method):
    """
    compute_foregrounds()

    Function to compute all foregrounds ...
    """

    # for each query image, apply the corresponding mask
    for image_filename in sorted(os.listdir(images_path)):
        if image_filename.endswith('.jpg'):
            image_path = os.path.join(images_path, image_filename)
            mask_path = os.path.join(masks_path,image_filename.split(".")[0]+ '.png')
            
            foreground = get_foreground(image_path, mask_path)

            cv.imwrite(os.path.join(masks_path,image_filename.split(".")[0])+ '.jpg',foreground)
            
            
            