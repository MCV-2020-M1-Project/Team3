import sys

from week1 import run_w1 as w1
from week2 import run_w2 as w2
from week3 import run_w3 as w3
from week4 import run_w4 as w4
import cv2 as cv
from week4 import feature_descriptors as fd
from week4 import utils as u
if __name__=="__main__":

    if "--week1" in sys.argv:
        w1.run()

    if "--week2" in sys.argv:
        w2.run()

    if "--week3" in sys.argv:
        w3.run()

    if "--week4" in sys.argv:
        w4.run()



