from collections import namedtuple
import numpy as np
import cv2 as cv

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tlx, tly, brx, bry], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou


# def mean_iou(groundtruth_bboxes, predicted_bboxes):

examples = [
	Detection("/home/oscar/Desktop/image_0002.jpg", [[39, 63, 203, 112]], [[54, 66, 198, 114]]),
	Detection("/home/oscar/Desktop/image_0016.jpg", [[49, 75, 203, 125]], [[42, 78, 186, 126]]),
    Detection("/home/oscar/Desktop/image_0120.jpg", [[35, 51, 196, 110]], [[36, 60, 180, 108]]),
    Detection("/home/oscar/Desktop/image_0090.jpg", [[349, 84, 498, 133], [54, 79, 217, 139]], [[353, 89, 501, 138], [51, 83, 209, 129]])]

for detection in examples:
    mean_iou = 0
	# load the image
    image = cv.imread(detection.image_path)

    for idx in range(len(detection.gt)):
        gt_bbox = detection.gt[idx]
        pred_bbox = detection.pred[idx]

        # draw the ground-truth bounding box along with the predicted
        # bounding box
        cv.rectangle(image, tuple(gt_bbox[:2]), tuple(gt_bbox[2:]), (0, 255, 0), 2)
        cv.rectangle(image, tuple(pred_bbox[:2]), tuple(pred_bbox[2:]), (0, 0, 255), 2)

        # compute the intersection over union and display it
        iou = bbox_iou(gt_bbox, pred_bbox)
        mean_iou += iou

    mean_iou = mean_iou/float(len(detection.gt))

    cv.putText(image, "IoU: {:.4f}".format(mean_iou), (10, 30),
        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print("{}: {:.4f}".format(detection.image_path, mean_iou))
    # show the output image
    cv.imshow("Image", image)
    cv.waitKey(0)

    # return mean_iou
