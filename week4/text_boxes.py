import cv2 as cv

def detect_text_box(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    laplacian = cv.convertScaleAbs(laplacian)
    cnt = cv.findContours(laplacian, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    area_imagen = image.shape[0] * image.shape[1]

    xm = 0
    ym = 0
    wm = 0
    hm = 0

    area_max = 0
    for c in cnt[0]:
        x,y,w,h = cv.boundingRect(c)
        area = w*h
        # area = cv.contourArea(c)
        if area > 1000 and ((area / area_imagen) * 100 < 25):
            x, y, w, h = cv.boundingRect(c)
            if area > area_max:
                area_max = area
                xm, ym, wm, hm = x, y, w, h

    if area_max == 0:
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
        sobel = sobely + sobelx
        sobel = cv.convertScaleAbs(sobel)
        sobel = (255 - sobel)
        sobel = cv.GaussianBlur(sobel, (3, 3), 0)
        cnt2 = cv.findContours(sobel, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        area_max = 0
        for c in cnt2[0]:
            x, y, w, h = cv.boundingRect(c)
            area = h * w
            if area > 500 and ((area / area_imagen) * 100) < 20:
                if area > area_max:
                    area_max = area
                    xm, ym, wm, hm = x, y, w, h
    cv.rectangle(image, (xm, ym), (xm + wm, ym + hm), (36, 255, 12), 3)

    return [xm,ym,xm + wm,hm + ym]

# PROBLEM!!!!! TEXT BOXES COORDINATES
def remove_text(paintings, paintings_coords):
    text_boxes = []
    for painting_id, painting in enumerate(paintings):
        [tlx, tly, brx, bry] = detect_text_box(painting)

        # If there is more than one painting, when detecting the text bouning box
        # we have to shift the coordinates so that they make sense in the initial image
        tlx += paintings_coords[painting_id][0]
        tly += paintings_coords[painting_id][1]
        brx += paintings_coords[painting_id][0]
        bry += paintings_coords[painting_id][1]

        text_boxes.append([tlx, tly, brx, bry])

    return [paintings, text_boxes]
