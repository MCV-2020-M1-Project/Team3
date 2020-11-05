import cv2 as cv


def surf_descriptor(image, threshold=400):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create(threshold, )
    kp, des = surf.detectAndCompute(img_gray, None)
    features=[]

    for k, d in (zip(kp,des)):
        features.append([k.pt, d.tolist()])

    return features