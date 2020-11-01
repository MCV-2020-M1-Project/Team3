from skimage import feature
import numpy as np
from scipy.fftpack import dct, idct
import cv2 as cv
from matplotlib import pyplot as plt
import math
from numpy import pi
from numpy import r_
from scipy.stats import itemfreq
import pywt

def lbp_hist(image, numPoints=16, radius=2, eps=1e-7):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")

    hist = cv.calcHist([lbp.astype(np.uint8)], [0], None, [numPoints + 2], [0, numPoints + 2])

    return hist

def dct_hist(image, norm='ortho', n_coefs=10):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    dct_img = dct2(gray, norm)

    dct_zigzag = np.asarray(zigzag(dct_img, n_coefs))

    hist = cv.calcHist([dct_zigzag.astype(np.uint16)], [0], None, [8], [0, max(dct_zigzag)])

    hist = hist.astype("float32")

    return hist

def dct2(image, norm='ortho'):
    return dct(dct(image, axis=0, norm=norm), axis=1, norm=norm)

def idct2(image, norm='ortho'):
    return idct(idct(image, axis=0, norm=norm), axis=1, norm=norm)

def zigzag(input, n_coefs):
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]

    output = []

    while (v < vmax) and (h < hmax) and (len(output) < n_coefs):
        output.append(input[v,h])

        if ((h + v) % 2) == 0:                 # going up

def hog(image):
    hog = cv.HOGDescriptor()
    return hog.compute(image)

def wavelet(image, mode='haar', level=1):
    image = np.float32(image)
    image /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(image, mode, level=level)

    # Process Coefficients
    # Here we need to do something with the coeficients returned but I'm not sure what to do
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0 #remove this line for the original image given to show after reconstruction

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    # Display result
    cv.imshow('image', imArray_H)
    cv.waitKey(0)

    return coeffs

            if (v == vmin):

                if (h == hmax-1):
                    v = v + 1

                else:
                    h = h + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                v = v + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                v = v - 1
                h = h + 1

        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                h = h + 1

            elif (h == hmin):                  # if we got to the first column
                if (v == vmax -1):
                    h = h + 1

                else:
                    v = v + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                v = v + 1
                h = h - 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output.append(input[v, h])
            break

    return output


def inverse_zigzag(input, vmax, hmax):

    h = 0
    v = 0

    hmin = 0
    vmin = 0

    output = np.zeros((vmax, hmax))

    i = 0

    while True:
        if ((h + v) % 2) == 0:                 # going up
            if (v == vmin):
                #print(1)
                output[v, h] = input[i]        # if we got to the first line

                if (h == hmax-1):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1

        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[v, h] = input[i]
                if (v == vmax-1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1

            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)
            output[v, h] = input[i]
            break

        # print(output)
        # print('h, v, i: {}, {}, {}'.format(h, v, i))
        # print('-------')

    return output
