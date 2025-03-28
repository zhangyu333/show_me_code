import cv2
import numpy as np
import onnxruntime as ort
import onnxruntime

onnxruntime.set_default_logger_severity(3)


class FastFaceDetect:
    def __init__(self):
        self.model = ort.InferenceSession("models/fast_face_detect.onnx")

    def imPreProc(self, img_path):
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
        else:
            img = img_path
        h, w, _ = img.shape
        img_resize = cv2.resize(img, (320, 240))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        img_resize = img_resize - 127.0
        img_resize = img_resize / 128.0
        img_resize = img_resize.transpose((2, 0, 1))
        img_resize = img_resize[None]
        return img, img_resize.astype("float32"), h, w

    def areaOf(self, left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iouOf(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.areaOf(overlap_left_top, overlap_right_bottom)
        area0 = self.areaOf(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.areaOf(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def hardNms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iouOf(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]

    def predict(self, width, height, confidences, boxes, prob_threshold=0.7, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self.hardNms(box_probs,
                                     iou_threshold=iou_threshold,
                                     top_k=top_k,
                                     )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def __call__(self, filename):
        org_image, input_img, height, width = self.imPreProc(filename)
        confidences, boxes = self.model.run(None, {"input": input_img})
        boxes, labels, probs = self.predict(width, height, confidences, boxes)
        return {"boxes": boxes, "labels": labels, "score": probs}
