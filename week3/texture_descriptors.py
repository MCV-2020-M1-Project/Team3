from skimage import feature
import numpy as np
from scipy.fftpack import dct, idct
import cv2 as cv
import pywt

def lbp_hist(image, numPoints, radius, eps=1e-7):

    lbp = feature.local_binary_pattern(image, numPoints,
                                       radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    hist /= (hist.sum() + eps)
    print(hist)
    return hist

def dct2(image, norm='ortho'):
    return dct(dct(image.T, norm=norm).T, norm=norm)

def idct2(image, norm='ortho'):
    return idct(idct(image.T, norm=norm).T, norm=norm)

def zigzag(image):

    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)

    xs = range(image)
    return {index: n for n, index in enumerate(sorted(((x, y) for x in xs for y in xs), key=compare))}

def dct_descriptor(image, norm, n_coefs):

    dct = dct2(image, norm)
    zig_zag = zigzag(dct)
    coefs = zig_zag[:n_coefs]

    return coefs

def hog(image):
    hog = cv.HOGDescriptor()
    return hog.compute(image)

def wavelet(image, mode='haar', level=1):
    coeffs = pywt.wavedec2(image, mode, level=level)
    return list(coeffs)
