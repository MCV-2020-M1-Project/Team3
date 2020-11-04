from week4 import sift


def get_corners():
    query_path = 'data/qsd1_w4'

    image_path = query_path + '/00000.jpg'

    sift.sift_corner_detection(image_path)


def run():
    get_corners()