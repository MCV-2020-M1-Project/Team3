import cv2 as cv

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

    bg_mask_painting = np.zeros(bg_mask.shape, dtype=np.uint8)
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

        if len(paintings_coords) > max_paintings:
            break

    return [mask, paintings_coords]

def remove_bg(img, params, image_id):
    [mask, paintings_coords] = get_bg_mask(img, params['remove'].max_paintings)

    result_bg_path = os.path.join(params['paths'].results, image_id + '.png')
    cv.imwrite(result_bg_path, mask)

    paintings = []
    for painting_id, painting_coords in enumerate(paintings_coords)
        result_painting_path = os.path.join(params['paths'].results, image_id + '_' + painting_id + '.jpg')
        paintings.append(get_painting_from_mask(img, mask, painting_coords, result_painting_path))

    return [paintings, paintings_coords]
