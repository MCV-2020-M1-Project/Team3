# -*- coding: utf-8 -*-
"""
by: Ian Riera Smolinska

Project M1 - Week 1
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def get_histogram(image,space_color):   
    if space_color == 'gray':
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist([gray],[0],None,[256],[0,256]) # consider size of the bins
    elif space_color == 'rgb':
        # to-do: way to concatenate the separate channels
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) #imread default columns are BGR instead of RGB
        #cv.calcHist(image, [channel], None, [histSize], [histRange])
        hist = cv.calcHist([rgb],[0,1,2], None, [256, 256, 256], [0, 256, 0, 256,0, 256] ) 
               
    else:
        # to do: change for exception
        raise Exception("Invalid Space Colour Provided")
        
    return hist


def plot_histogram(hist):
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()
    # to-do multiple channels case
