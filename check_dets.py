import cv2
from ultralytics import YOLO

model = YOLO("assets/models/epoch40.pt")
cap = cv2.VideoCapture("assets/sample_video/drone_3.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video FPS: {fps}, Size: {width}x{height}")

# Go to 4 seconds (approx 120 frames)
target_frame = int(4 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx = target_frame + i
    # Predict with low confidence threshold to see if pedestrian is detected
    results = model(frame, conf=0.01, verbose=False)
    boxes = results[0].boxes
    
    print(f"\n--- Frame {frame_idx} ---")
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().tolist()
            # If the box is in the bottom-right region (e.g., x > width * 0.7, y > height * 0.7)
            if xyxy[0] > width * 0.7 and xyxy[1] > height * 0.7:
                print(f"Box: {xyxy}, Class: {cls} ({model.names[cls]}), Conf: {conf:.4f}")

cap.release()
