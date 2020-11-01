import os
import cv2
import numpy as np
from math import log10, sqrt
from scipy import stats


def readImages(path,extension):
    imagenes = dict()
    for filename in sorted(os.listdir(path)):
        if filename.find(extension) != -1:
            img = cv2.imread(os.path.join(path, filename))
            imagenes[filename[0:-4]] = img
    return imagenes

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def denoiseImage(img):
    gaussiana =  cv2.medianBlur(img,3)
    psnr = PSNR(img, gaussiana)
    if psnr < 33:
        img = gaussiana
    
    return img


"""path = "../data/qsd1_w3"
imagenes = readImages(path,".jpg")

for img in imagenes:
    imagenes[img] = denoiseImage(imagenes[img],img)
    #cv2.imshow(img,imagenes[img])
    cv2.imwrite("../data/filtradas/"+img+".jpg",imagenes[img])
cv2.waitKey()"""