import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def run():
    # Convert images to matrix
    img_1 = cv.imread("qsd1_w1/00000.jpg")
    img_2 = cv.imread("qsd1_w1/00001.jpg")

    space_color = "gray"

    # Task1

    # calculate histograms
    hist_1 = histogram.get_histogram(img_1,space_color)
    hist_2 = histogram.get_histogram(img_2,space_color)

    # calculate histograms
    histogram.plot_histogram(hist_1)
    histogram.plot_histogram(hist_2)

    # Task 2

    # compare histograms - to define function that calculates diferent distances and returns array of values
    intersection = cv.compareHist(hist_1,hist_2,cv.HISTCMP_INTERSECT)
    print(intersection)


    # Use CV to convert images