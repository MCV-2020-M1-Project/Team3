# boxes = pickle.load(open(os.path.join(boxes_path), 'rb'))[image_id][0]

from collections import namedtuple
import numpy as np
import cv2 as cv
import pickle

def box_iou(boxA, boxB):
    # compute the intersection over union of two boxes

    # Format of the boxes is [tlx, tly, brx, bry], where tl and br
    # indicate top-left and bottom-right corners of the box respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def mean_iou(gt_boxes_path, pred_boxes_path):

    # image = cv.imread('/home/oscar/workspace/master/modules/m1/project/Team3/data/qsd1_w2/00005.jpg')

    all_gt_boxes = pickle.load(open(gt_boxes_path, 'rb'))
    all_pred_boxes = pickle.load(open(pred_boxes_path, 'rb'))

    total_boxes = 0
    mean_iou = 0

    all_gt_boxes_corrected = []

    # Groundtruth of qsd1_2 in wrong format --> change it
    if 'qsd1_w2' in gt_boxes_path:
        for img_gt_boxes in all_gt_boxes:
            boxes_per_image = []
            for gt_box in img_gt_boxes:
                tlx = gt_box[0][0]
                tly = gt_box[0][1]
                brx = gt_box[2][0]
                bry = gt_box[2][1]
                boxes_per_image.append([tlx,tly,brx,bry])
            all_gt_boxes_corrected.append(boxes_per_image)
    else:
        all_gt_boxes_corrected = all_gt_boxes

    # Get gt bounding boxes of each image
    for img_idx, img_gt_boxes in enumerate(all_gt_boxes_corrected):
        img_pred_boxes = all_pred_boxes[img_idx]
        # image = cv.imread(images_paths[img_idx])

        gt_num_boxes = len(img_gt_boxes)
        pred_num_boxes = len(img_pred_boxes)

        # If there are detected boxes
        if pred_num_boxes > 0:
            # Compute iou for each gt/predicted bounding box
            for box_idx, gt_box in enumerate(img_gt_boxes):

                pred_box = img_pred_boxes[box_idx]

                # # draw the ground-truth bounding box along with the predicted
                # # bounding box
                # cv.rectangle(image, tuple(gt_box[:2]), tuple(gt_box[2:]), (0, 255, 0), 2)
                # cv.rectangle(image, tuple(pred_box[:2]), tuple(pred_box[2:]), (0, 0, 255), 2)

                # compute the intersection over union and display it

                iou = box_iou(gt_box, pred_box)
                mean_iou += iou

                # cv.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
                #     cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                #
                # # show the output image
                # cv.imshow("Image", image)
                # cv.waitKey(0)

                if pred_num_boxes == 1:
                    break

        total_boxes += gt_num_boxes

    mean_iou = mean_iou/float(total_boxes)

    print("{:.4f}".format(mean_iou))

    return mean_iou
