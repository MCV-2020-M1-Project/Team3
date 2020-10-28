from skimage import feature
import numpy as np
import cv2 as cv

def lbp_hist(image, numPoints, radius, eps=1e-7):

    lbp = feature.local_binary_pattern(image, numPoints,
                                       radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    hist /= (hist.sum() + eps)
    print(hist)
    return hist

