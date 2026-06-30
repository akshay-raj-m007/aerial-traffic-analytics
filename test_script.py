import cv2
import numpy as np
from core.detector import Detector

detector = Detector()
cap = cv2.VideoCapture("assets/sample_video/drone_3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

target_frame = int(4 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, frame = cap.read()
if ret:
    dets = detector.detect(frame, frame_idx=target_frame)
    for d in dets:
        print(d)
else:
    print("Could not read frame")
cap.release()
