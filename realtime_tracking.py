from __future__ import division, print_function, absolute_import
from detector import Detector
from encoder import Encoder
from tracker import Tracker
import numpy as np
import cv2


class Person(object):
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


def tracking(video_path):

    detector = Detector('./yolov3_model/model.ckpt')
    encoder = Encoder('./mars_model/model.pb')
    tracker = Tracker("cosine", 0.3)

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        _, frame = cap.read()

        scores, boxes, box_classes = detector.detect(frame)
        features = encoder.encode(frame, boxes, 32)
        persons = [Person(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]
        tracker.predict()
        tracker.update(persons)

        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        cv2.imshow('', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    tracking(0)
