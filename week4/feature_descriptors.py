import cv2 as cv


def surf_descriptor(image, threshold=400):
    cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print('surf')
    surf = cv.xfeatures2d.SURF_create(threshold, )
    kp, des = surf.detectAndCompute(image, None)
    features=[]

    for k, d in (zip(kp,des)):
        features.append([k.pt, d.tolist()])

    return features