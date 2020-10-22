from week2 import text_finder as tf
import cv2 as cv

def run():
    image_path='D:\MCV\M1\Project\qsd2_w2\\00000.jpg'
    mask = tf.get_mask(image_path)
    cv.imwrite("D:\MCV\M1\Project\qsd1_w2\\00000.png", mask)
