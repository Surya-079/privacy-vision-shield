import cv2
import torch
import numpy as np
from ultralytics import YOLOWorld

# --- SYLLABUS TOPIC 1: Open-Vocabulary Deep Learning Network ---
device = "cpu"
print(f"Loading Next-Gen Open-Vocabulary Model on: {device}")

# Load a pre-trained YOLO-World model capable of parsing arbitrary text prompts
model = YOLOWorld("yolov8s-world.pt")

# Set dynamic target anchors. This tells the network exactly what to look for, 
# allowing it to catch small visiting cards even when partially occluded by fingers.
model.set_classes(["card", "visiting card", "business card", "paper rectangle"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access physical host webcam hardware layer.")
    exit()

print("\n>>> Syllabus-Compliant YOLO-World Tracker Active.")
print(">>> Show any card (held by fingers or angled). Press q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    h, w, _ = frame.shape
    
    # Run the open-vocabulary deep inference detection tracker
    results = model.predict(frame, conf=0.15, verbose=False)
    
    if results and len(results) > 0:
        frame_result = results[0]
        
        if frame_result.boxes is not None:
            for box in frame_result.boxes:
                # Extract positional scalar coordinates out of the tensor array maps
                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                # Enforce visual border clipping constraints safely
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                # --- SYLLABUS TOPIC 2: Region of Interest (ROI) Pixel Mining ---
                # Calculate bounding box area metrics
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Validate area dimensions to ignore random microscopic artifacts
                if box_w * box_h < 3000:
                    continue
                
                # Isolate the text-bearing window region of the card
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Apply an intense mathematical Gaussian kernel blur over the card surface
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (99, 99), 0)
                    
                    # Draw compliant tracking borders that follow the card movement fluidly
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, "PRIVACY PROTECTION", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Output the live visualization window directly onto your desktop workspace
    cv2.imshow("Privacy Shield - Live Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("System execution streams closed gracefully.")
