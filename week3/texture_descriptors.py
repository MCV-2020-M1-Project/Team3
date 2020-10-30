from skimage import feature
import numpy as np
from scipy.fftpack import dct, idct
import cv2 as cv
from matplotlib import pyplot as plt
import math
from numpy import pi
from numpy import r_
from scipy.stats import itemfreq

def lbp_hist(image, numPoints=24, radius=3, eps=1e-7):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")

    hist = cv.calcHist([lbp.astype(np.uint8)], [0], None, [numPoints + 2], [0, numPoints + 2])

    return hist

def dct_hist(image, norm='ortho', n_coefs=10):

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    dct_img = dct2(image, norm)

    dct_zigzag = np.asarray(zigzag(dct_img))

    dct_zigzag = dct_zigzag[:n_coefs]

    hist = np.ascontiguousarray(itemfreq(dct_zigzag.ravel()))

    # print(hist)

    hist = hist.astype("float32")

    # print(dct_zigzag)
    # hist /= (hist.sum() + 1e-7)

    # print(hist)

    # dct_zigzag = dct_zigzag[:n_coefs]
    # print(dct_zigzag)

    # print(dct_zigzag.astype(np.uint8))

    # print(min(dct_zigzag))
    # print(max(dct_zigzag))

    # hist = cv.calcHist([dct_zigzag.astype(np.uint8)], [0], None, [8], [0, max(dct_zigzag)])

    # print(image.shape)
    # print(len(hist))
    #
    # print(hist[0].sum())
    #
    # print('-------')

    # dct_zigzag[-5:] = 0
    # dct_reverted = inverse_zigzag(dct_zigzag, dct_img.shape[0], dct_img.shape[1])
    # idct_img = idct2(dct_reverted)

    # coefs = dct_zigzag[:n_coefs]

    # print(coefs)

    return hist


def dct2(image, norm='ortho'):
    return dct(dct(image, axis=0, norm=norm), axis=1, norm=norm)

def idct2(image, norm='ortho'):
    return idct(idct(image, axis=0, norm=norm), axis=1, norm=norm)

def zigzag(img):

    [h,w] = img.shape
    zigzag=[[] for i in range(h+w-1)]

    for i in range(h):
        for j in range(w):
            sum=i+j
            if(sum%2 ==0):
                #add at beginning
                zigzag[sum].insert(0,img[i,j])
            else:

                #add at end of the list
                zigzag[sum].append(img[i,j])

    reordered = []
    for i in zigzag:
        for j in i:
            reordered.append(j)

    return reordered

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


# im = cv.imread('/home/oscar/workspace/master/modules/m1/project/Team3/data/qsd1_w2/00001.jpg')

# # plt.gray()
# # plt.subplot(131), plt.imshow(im), plt.axis('off'), plt.title('original image')
# # # plt.subplot(132), plt.imshow(im1), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)')
# # plt.subplot(132), plt.imshow(np.log(1e-7 + abs(lI))), plt.title('DCT')
# # plt.subplot(133), plt.imshow(np.log(1e-7 + abs(lI_thresh))), plt.title('DCT')
# # plt.show()

# matrix = np.matrix([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
# matrix_reordered = np.asarray(zigzag(matrix))
# matrix_reordered[-5:] = 0
# reverted = inverse_zigzag(matrix_reordered, matrix.shape[0], matrix.shape[1])
#
# print(matrix)
# print(matrix_reordered)
# print(reverted)

# imsize = im.shape
# dct_img = np.zeros(imsize)
#
# im_dct = np.zeros(imsize)
#
# # Do 8x8 DCT on image (in-place)
# for i in r_[:imsize[0]:8]:
#     for j in r_[:imsize[1]:8]:
#         dct_img[i:(i+8),j:(j+8)] = dct2(im[i:(i+8),j:(j+8)])
#
#         dct_block = dct_img[i:(i+8),j:(j+8)]
#
#         dct_img_reordered = np.asarray(zigzag(dct_block))
#
#         dct_img_reordered[-62:] = 0
#
#         reverted = inverse_zigzag(dct_img_reordered, dct_block.shape[0], dct_block.shape[1])
#
#         # thresh = 0.2
#         # dct_thresh = dct_block * (abs(dct_block) > (thresh*np.max(dct_block)))
#
#         im_dct[i:(i+8),j:(j+8)] = idct2(reverted)

# dct_img = inverse_zigzag(dct_img, 8, 8)

# # Threshold
# thresh = 0.1
# dct_thresh = dct_img * (abs(dct_img) > (thresh*np.max(dct_img)))


# plt.figure()
# plt.imshow(dct_thresh,cmap='gray',vmax = np.max(dct_img)*0.01,vmin = 0)
# plt.title( "Thresholded 8x8 DCTs of the image")

# percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)

# print("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros*100.0))

# im_dct = np.zeros(imsize)
#
# for i in r_[:imsize[0]:8]:
#     for j in r_[:imsize[1]:8]:
#         im_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )


# plt.figure()
# plt.imshow( np.hstack( (im, im_dct) ) ,cmap='gray')
# plt.title("Comparison between original and DCT compressed images" )
#
# plt.show()
