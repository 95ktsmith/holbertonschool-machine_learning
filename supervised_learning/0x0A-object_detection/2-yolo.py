#!/usr/bin/env python3
""" Yolo """
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Class to use Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used for the
            Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the initial
            filtering step
        nms_t is a float representing the IOU threshold for non-max suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            outputs is the number of outputs (predictions) made by the
                Darknet model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """ Returns the value of sigmoid(x) expression """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        outputs is a list of numpy.ndarrays containing the predictions from the
            Darknet model for a single image:
            Each output will have the shape (grid_h, grid_w,
                anchor_boxes, 4 + 1 + classes)
                grid_h & grid_w => the height and width of the grid
                    used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original size
            [image_height, image_width]

        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_h, grid_w,
                anchor_boxes, 4) containing the processed boundary boxes for
                each output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative
                    to original image
            box_confidences: a list of numpy.ndarrays of shape
                (grid_h, grid_w, anchor_boxes, 1) containing the box
                confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape
                (grid_h, grid_w, anchor_boxes, classes) containing
                the box’s class probabilities for each output, respectively
        """
        image_height, image_width = image_size
        boxes, box_confidences, box_class_probs = [], [], []

        for output in range(len(outputs)):
            grid_h, grid_w, anchor_boxes = outputs[output].shape[:3]
            classes = outputs[output].shape[3] - 5
            input_width = int(self.model.input.shape[1])
            input_height = int(self.model.input.shape[2])
            boxes_tmp = np.zeros((grid_h, grid_w, anchor_boxes, 4))
            conf_tmp = np.zeros((grid_h, grid_w, anchor_boxes, 1))
            prob_tmp = np.zeros((grid_h, grid_w, anchor_boxes, classes))
            for row in range(grid_h):
                for col in range(grid_w):
                    for box in range(anchor_boxes):
                        # Calculate top left and bottom right corners of each
                        # boundary box
                        tx, ty, tw, th = outputs[output][row][col][box][:4]
                        pw = self.anchors[output, box, 0]
                        ph = self.anchors[output, box, 1]

                        # Calculate center, width and height offsets
                        bx = self.sigmoid(tx) + col
                        by = self.sigmoid(ty) + row
                        bw = pw * np.exp(tw)
                        bh = ph * np.exp(th)
                        # Normalize
                        bx /= grid_w
                        by /= grid_h
                        bw /= input_width
                        bh /= input_height
                        # Calculate real positions
                        x1 = (bx - bw / 2) * image_width
                        x2 = (bx + bw / 2) * image_width
                        y1 = (by - bh / 2) * image_height
                        y2 = (by + bh / 2) * image_height
                        boxes_tmp[row][col][box] = np.array([x1, y1, x2, y2])

                        # Calculate confidences for each anchor box
                        pc = outputs[output][row][col][box][4]
                        conf_tmp[row, col, box, 0] = self.sigmoid(pc)

                        # Calculate class probabilities for each anchor box
                        class_probs = outputs[output][row][col][box][5:]
                        prob_tmp[row, col, box] = self.sigmoid(class_probs)

            boxes.append(boxes_tmp)
            box_confidences.append(conf_tmp)
            box_class_probs.append(prob_tmp)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters out boxes below confidence threshold
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the processed box
            confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the processed box
            class probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
                the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
                number that each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
                for each box in filtered_boxes, respectively
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for i in range(len(boxes)):
            probs = box_confidences[i][:, :, :, :] *\
                          box_class_probs[i][:, :, :, :]
            predictions = np.amax(probs, axis=3)
            kept = np.argwhere(predictions[:, :, :] >= self.class_t)

            for idx in kept:
                filtered_boxes.append(boxes[i][idx[0], idx[1], idx[2]])
                box_scores.append(predictions[idx[0], idx[1], idx[2]])
                box_classes.append(np.argmax(probs[idx[0], idx[1], idx[2]]))

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)
        return (filtered_boxes, box_classes, box_scores)
