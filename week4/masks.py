import os
import cv2 as cv
import numpy as np
from skimage import feature

import scipy.stats as stats
import week4.evaluation as evaluation

import week4.utils as utils

def get_painting_from_mask(img, bg_mask, painting_coords, result_painting_path):
    """
    get_painting_from_mask()

    Function to extract a painting from an image using the background mask ...
    """

    # Coordinates of the painting
    tlx = painting_coords[0]
    tly = painting_coords[1]
    brx = painting_coords[2]
    bry = painting_coords[3]

    bg_mask_painting = np.zeros(img.shape, dtype=np.uint8)
    bg_mask_painting[tly:bry, tlx:brx] = 255

    # Combine the image and the background mask
    combined = cv.bitwise_and(img, bg_mask_painting)

    gray_combined = cv.cvtColor(combined,cv.COLOR_BGR2GRAY)

    # Coordinates of non-black pixels.
    coords = np.argwhere(gray_combined)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    # Get the contents of the bounding box.
    painting = combined[x0:x1, y0:y1]

    # Save extracted painting
    cv.imwrite(result_painting_path, painting)

    return painting


def get_bg_mask1(image, max_paintings):
    """
    get_mask_M0()

    Function to compute a binary mask of the background using method 0...
    """

    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    h,s,v = cv.split(image_hsv)

    # 0s --> contours
    mask = cv.adaptiveThreshold(s, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv.THRESH_BINARY, 51, 10)

    # 1s --> contours
    mask = 255-mask

    # Denoising with "opening" morphology operator
    dilatation_size = 1
    element = cv.getStructuringElement(cv.MORPH_RECT, (2*dilatation_size+1, 2*dilatation_size+1),
                                        (int(dilatation_size/2), int(dilatation_size/2)))
    mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, element, iterations=3)

    # Coordinates of non-black pixels (picture contours)
    coords = np.argwhere(mask_open)

    # First and last non-black pixel
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)

    # Bounding box of non-black pixels
    pnts = np.asarray([[y0,x0], [y0,x1], [y1,x1], [y1,x0]], dtype=np.int32)
    final_mask = np.zeros(mask.shape)
    cv.fillConvexPoly(final_mask, pnts, 255)

    return [final_mask, [[y0,x0,y1,x1]]]


def get_bg_mask(img, max_paintings):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = feature.canny(gray, sigma=3,low_threshold=20,high_threshold=40)
    mask = np.zeros(gray.shape)

    mask_copy=mask.copy()

    mask_copy[edges]=255
    mask_copy=cv.convertScaleAbs(mask_copy)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(30,30))
    closed = cv.morphologyEx(mask_copy, cv.MORPH_CLOSE, kernel)

    cnts = cv.findContours(closed.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    mask = np.zeros(gray.shape)
    paintings_coords = []

    # loop over the contours from bigger to smaller, and find the biggest one with the right orientation
    for c in cnts:
        # approximate to the rectangle
        x, y, w, h = cv.boundingRect(c)

        if w > gray.shape[1]/8 and h > gray.shape[0]/6:
            mask[y:y+h,x:x+w] = 255 # fill the mask
            paintings_coords.append([x,y,x+w,y+h])

        if len(paintings_coords) == max_paintings:
            break

    return [mask, paintings_coords]

def remove_bg(img, params, image_id):
    #We need to fix it
    # [mask, paintings_coords] = get_bg_mask(img, params['remove']['max_paintings'])
    [mask, paintings_coords] = get_bg_mask_g7(img)

    result_bg_path = os.path.join(params['paths']['results'], image_id + '.png')
    cv.imwrite(result_bg_path, mask)

    # print(f'{image_id} --> {paintings_coords}')

    paintings = []
    for painting_id, painting_coords in enumerate(paintings_coords):
        result_painting_path = os.path.join(params['paths']['results'], image_id + '_' + str(painting_id) + '.jpg')
        paintings.append(get_painting_from_mask(img, mask, painting_coords, result_painting_path))

    return [paintings, paintings_coords]



# ------------------- TEAM 7 CODE -------------------

def get_bg_mask_g7(image):
    """ Obtain a mask for each image in the list of images query_img. Method: PBM.
        params:
            query_imgs: List of images of the query set
        returns:
            List of masks [2D images with 1 channel]. A pixel = 0 means background, and pixel = 255 painting
    """
    # print("Obtaining masks")

    [mask, paintings_coords] = pbm_segmentation(image)

    # mask = fill_mask(mask, paintings_coords)

    return [mask, paintings_coords]

def pbm_segmentation(img, margin=0.02, threshold=0.000001):
    """ Probability-based segmentation. Model the background with a multivariate gaussian distribution.
            img: Image to be segmented
            margin: Part of image being certainly part of the background
            threshold: Threshold to reject a pixel as not being part of the background
        returns: Mask image, indicating the painting part in the image
    """

    img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    # Mask based on a bivariate gaussian distribution
    mask = compute_mask_gaussian_HSL(img, margin, threshold)

    # Compute mask based on connected components
    [resulting_mask, paintings_coords] = mask_segmentation_cc(img, mask)

    final_paintings_coords = []

    if len(paintings_coords) == 2:
        tlx1 = paintings_coords[0][1]
        tly1 = paintings_coords[0][0]
        brx1 = paintings_coords[0][3]
        bry1 = paintings_coords[0][2]

        tlx2 = paintings_coords[1][1]
        tly2 = paintings_coords[1][0]
        brx2 = paintings_coords[1][3]
        bry2 = paintings_coords[1][2]

        if (tlx1 < tlx2 and brx1 < tlx2) or (tly1 < tly2 and bry1 < tly2):
            final_paintings_coords.append([tlx1, tly1, brx1, bry1])
            final_paintings_coords.append([tlx2, tly2, brx2, bry2])
        else:
            final_paintings_coords.append([tlx2, tly2, brx2, bry2])
            final_paintings_coords.append([tlx1, tly1, brx1, bry1])
    else:
        final_paintings_coords.append([paintings_coords[0][1], paintings_coords[0][0],
                                       paintings_coords[0][3], paintings_coords[0][2]])

    return [resulting_mask, final_paintings_coords]

def fill_mask(mask, paintings_coords):
    filled_mask = np.zeros(mask.shape, dtype=mask.dtype)
    for painting_coords in paintings_coords:
        tlx = painting_coords[0]
        tly = painting_coords[1]
        brx = painting_coords[2]
        bry = painting_coords[3]
        filled_mask[tly:bry,tlx:brx] = 255

    return filled_mask

def mask_segmentation_cc(img, mask):

    kernel = np.ones((img.shape[0]//50, img.shape[1]//50), np.uint8)
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, kernel, borderValue=0)

    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    sizes = stats[:, -1]

    top_two_conn_comp_idx = sizes.argsort()
    top_two_conn_comp_idx = top_two_conn_comp_idx[top_two_conn_comp_idx!=0]
    if len(top_two_conn_comp_idx) > 1:
        top_two_conn_comp_idx = top_two_conn_comp_idx[[-2,-1]][::-1]
    else:
        top_two_conn_comp_idx = top_two_conn_comp_idx[[-1]][::-1]

    idxs = [idx for idx in top_two_conn_comp_idx]

    bc = np.zeros(output.shape)
    bc[output == idxs[0]] = 255
    # bc = create_convex_painting(mask, bc)

    if len(idxs) > 1:
        sbc = np.zeros(output.shape)
        sbc[output == idxs[1]] = 255
        sbc = create_convex_painting(mask, sbc)

    paintings_coords = [get_bbox(bc)]
    resulting_mask = bc

    # Second painting if first one does not take most part + more or less a rectangular shape + no IoU
    if len(idxs) > 1:
        if not takes_most_part_image(bc) and regular_shape(sbc) and check_no_iou(bc, sbc):
            paintings_coords.append(get_bbox(sbc))
            resulting_mask = np.logical_or(resulting_mask==255, sbc==255).astype(np.uint8)*255

    return [resulting_mask, paintings_coords]

def create_convex_painting(mask, component_mask):
    kernel = np.ones((5, 5), np.uint8)
    component_mask = cv.morphologyEx(component_mask, cv.MORPH_CLOSE, kernel, borderValue=0)
    _,contours, hierarchy = cv.findContours((component_mask == 255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask).astype(np.uint8)
    polished_mask = cv.fillPoly(mask, contours, 255).astype(np.uint8)
    a = polished_mask.copy()

    p = int(max(mask.shape[0]/8, mask.shape[1]/8))
    polished_mask = cv.copyMakeBorder(src=polished_mask, top=p, bottom=p, left=p, right=p, borderType=cv.BORDER_CONSTANT, value=0)
    size1, size2 = int(mask.shape[0]*1/32),int(mask.shape[1]*1/32)
    kernel = np.ones((size1, size2), np.uint8)
    polished_mask = cv.morphologyEx(polished_mask, cv.MORPH_CLOSE, kernel, borderValue=0)
    size1, size2 = int(mask.shape[0]/8), int(mask.shape[1]/8)
    kernel = np.ones((size1, size2), np.uint8)
    polished_mask = cv.morphologyEx(polished_mask, cv.MORPH_OPEN, kernel, borderValue=0)

    if len(polished_mask[polished_mask!=0]) != 0:
        rect_portion = 0.6
        x0,y0,x1,y1 = get_bbox(polished_mask)
        kernel = np.ones((int((x1-x0)*rect_portion), int((y1-y0)*rect_portion)), np.uint8)
        polished_mask = cv.morphologyEx(polished_mask, cv.MORPH_OPEN, kernel, borderValue=0)
    return polished_mask[p:polished_mask.shape[0]-p, p:polished_mask.shape[1]-p]

def takes_most_part_image(img):
    h_quarter, w_quarter = img.shape[0]//4, img.shape[1]//4
    return img[h_quarter, w_quarter*2] == 1 and img[h_quarter*2, w_quarter] == 1 and img[h_quarter*3, w_quarter*2] == 1 and img[h_quarter*2, w_quarter*3] == 1

def get_bbox(mask):
    num_pixel_estimation = 20
    positions = np.where(mask==255)
    hs, ws = sorted(positions[0]), sorted(positions[1])
    h_min, h_max = int(np.array(hs[:num_pixel_estimation]).mean()), int(np.array(hs[-num_pixel_estimation:]).mean())
    w_min, w_max = int(np.array(ws[:num_pixel_estimation]).mean()), int(np.array(ws[-num_pixel_estimation:]).mean())
    return [h_min, w_min, h_max, w_max]

def regular_shape(mask, threshold=0.7):
    if mask.sum() == 0:
        return False
    h_min, w_min, h_max, w_max = get_bbox(mask)
    sum_pixels = (mask[h_min:h_max, w_min:w_max]==255).astype(np.uint8).sum()
    return sum_pixels/((h_max-h_min)*(w_max-w_min)) > threshold

def check_no_iou(mask1, mask2):
    bbox1, bbox2 = get_bbox(mask1), get_bbox(mask2)
    return evaluation.box_iou(bbox1, bbox2) < 1e-6

def compute_mask_gaussian_HSL(img, margin, threshold=0.000001):
    h_m, w_m = int(img.shape[0]*margin), int(img.shape[1]*margin)

    # Compute mean and standard deviation for each channel separately
    l_mean = (np.concatenate([img[:h_m, :, 0].reshape(-1), img[:, :w_m, 0].reshape(-1), img[img.shape[0]-h_m:, :, 0].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 0].reshape(-1)])).mean()
    a_mean = (np.concatenate([img[:h_m, :, 1].reshape(-1), img[:, :w_m, 1].reshape(-1), img[img.shape[0]-h_m:, :, 1].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 1].reshape(-1)])).mean()
    b_mean = (np.concatenate([img[:h_m, :, 2].reshape(-1), img[:, :w_m, 2].reshape(-1), img[img.shape[0]-h_m:, :, 2].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 2].reshape(-1)])).mean()

    l_std = (np.concatenate([img[:h_m, :, 0].reshape(-1), img[:, :w_m, 0].reshape(-1), img[img.shape[0]-h_m:, :, 0].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 0].reshape(-1)])).std()
    a_std = (np.concatenate([img[:h_m, :, 1].reshape(-1), img[:, :w_m, 1].reshape(-1), img[img.shape[0]-h_m:, :, 1].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 1].reshape(-1)])).std()
    b_std = (np.concatenate([img[:h_m, :, 2].reshape(-1), img[:, :w_m, 2].reshape(-1), img[img.shape[0]-h_m:, :, 2].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 2].reshape(-1)])).std()

    # Model background and discard unlikely pixels
    mask = stats.norm.pdf(img[:,:,0], l_mean, l_std)*stats.norm.pdf(img[:,:,1], a_mean, a_std)*stats.norm.pdf(img[:,:,2], b_mean, b_std) < threshold

    return mask

# ------------------- TEAM 7 CODE -------------------
