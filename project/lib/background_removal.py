
# import the necessary packages
import os
import imutils
import cv2
import numpy as np

# function that removes the background by searching the edges and contours
def remove_background(image_path):
    # Tunning parameters. We can put this as input to the function as well
    CANNY_THRESH_1 = 30
    CANNY_THRESH_2 = 130
    
    # load the input image
    image = cv2.imread(image_path,0)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # obtain the edges of the image
    edges = cv2.Canny(blurred, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    
    # find contours in the edged image
    cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
   
    # sort from biggest area to smallest and take the top5
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    
    
    mask = np.zeros(edges.shape)
    cmax, max_extent=[],0
    # loop over the contours from bigger to smaller, and find the biggest one with the right orientation
    for c in cnts:
          # # approximate to the hull.
          hull = cv2.convexHull(c)
        
          # find the contour with the highest extent compared to the bounding rectangle
          area = cv2.contourArea(hull)
          x,y,w,h = cv2.boundingRect(c)
          rect_area = w*h
          extent = float(area)/rect_area 


          # get the contour with max extent (area covered, approximation area)
          if max_extent<extent:
              max_extent=extent
              cmax=hull
         
    cv2.fillConvexPoly(mask, cmax, (255)) # fill the mask 
    
    return mask

def mask_evaluation(mask_path,annotation_path):
    mask = cv2.imread(mask_path,0)/255
    annotation = cv2.imread(annotation_path,0)/255

    
    mask_vector = mask.reshape(-1)
    annotation_vector = annotation.reshape(-1)
    
    TP = np.dot(annotation_vector,mask_vector) # true positive
    FP = np.dot(1 - annotation_vector,mask_vector) # false positive
    TN = np.dot(1 - annotation_vector,1 - mask_vector) # true negative
    FN = np.dot(annotation_vector,1 - mask_vector) # false positive
    
    P = TP/(TP+FP) # precision
    R = TP/(TP+FN) # recall
    F1 = 2*(P*R)/(P+R) # F1-score
    
    return P, R, F1


if __name__ == '__main__':
    """
    ......
    """
    
    qsd2_path = 'data/qsd2_w1/'
    qsd2_filenames = sorted(os.listdir(qsd2_path))
    
    # for each query image, find the corresponding mask
    for idx,qsd2_filename in enumerate(qsd2_filenames):
        if qsd2_filename.endswith('.jpg'):
            mask = remove_background(os.path.join(qsd2_path, qsd2_filename))
            cv2.imwrite(os.path.join(qsd2_path,qsd2_filename.split(".")[0])+'_G3.png',mask)

    # evaluate masks obtained
    for idx,qsd2_filename in enumerate(qsd2_filenames):
        if qsd2_filename.endswith('_G3.png'):
            P,R,F1 = mask_evaluation(os.path.join(qsd2_path, qsd2_filename),os.path.join(qsd2_path, qsd2_filename.split("_")[0]+'.png'))
            print(" Mask #{} --> Precision: {}, Recall: {}, F1-score: {}".format(qsd2_filename.split("_")[0], P,R,F1))

