import os
import imutils
import cv2 as cv
import numpy as np

def get_mask_M0(image):
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

    return final_mask

def get_mask_M1(image, color_space="RGB"):
    """
    get_mask_M1()

    Function to compute a binary mask of the background using method 1...
    """

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

def get_mask_M2(image, color_space="RGB"):
    """
    get_mask_M2()

    Function to compute a binary mask of the background using method 2...
    """

    mask = get_mask_M1(image, color_space)

    # Filling the holes with closing
    dilatation_size = 6
    element = cv.getStructuringElement(cv.MORPH_RECT, (2*dilatation_size+1, 2*dilatation_size+1),
                                        (int(dilatation_size/2), int(dilatation_size/2)))
    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, element, iterations=9)

    return mask_closed

def get_mask_M3(image):
    """
    get_mask_M3()

    Function to compute a binary mask of the background using method 3...
    """

    # Tunning parameters. We can put this as input to the function as well
    CANNY_THRESH_1 = 30
    CANNY_THRESH_2 = 130

    # load the input image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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

def get_mask_M4(image):
    # large = cv.imread(image_path)
    # rgb = cv.pyrDown(large)
    # small = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

    small = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # edges = cv.Canny(small,500,600)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) #or RECT or CROSS
    grad = cv.morphologyEx(small, cv.MORPH_GRADIENT, kernel)

    _, bw = cv.threshold(grad, 0.0, 255.0, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 10))
    connected = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    _, contours, hierarchy = cv.findContours(connected.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    final_contours = []

    found = False

    for idx in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.45 and w > small.shape[1]/4 and h > small.shape[0]/4:
            final_contours.append([x,y,w,h])
            cv.rectangle(image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

            found = True

    # print(small.shape)
    # print(final_contours)

    # cv.imshow('small', small)
    mask = np.zeros(small.shape)
    for box_coords in final_contours:
        x0 = box_coords[0]
        y0 = box_coords[1]
        x1 = box_coords[0] + box_coords[2]
        y1 = box_coords[1] + box_coords[3]

        pnts = np.asarray([[x0,y0], [x0,y1], [x1,y1], [x1,y0]], dtype=np.int32)
        cv.fillConvexPoly(mask, pnts, 255)
        # cv.imshow('mask',mask)
        # cv.waitKey(0)

    if not found:
        mask = get_mask_M0(image)

    # cv.imshow('small', small)
    # cv.waitKey(0)
    # cv.imshow('grad', grad)
    # cv.waitKey(0)
    # cv.imshow('bw', bw)
    # cv.waitKey(0)
    # cv.imshow('connected', connected)
    # cv.waitKey(0)
    # cv.imshow('rects', rgb)
    # cv.waitKey(0)

    # cv.fillConvexPoly()

    return mask
