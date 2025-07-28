import threading
import cv2
from deepface import DeepFace
from ultralytics import YOLO
import os

yolo_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

identified_person = "Unknown"
check_in_progress = False
counter = 0
frame_skip = 30
lock = threading.Lock()

def recognize_face(frame):
    global identified_person, check_in_progress
    try:
        result = DeepFace.find(img_path=frame, db_path="faces_db", enforce_detection=False)
        with lock:
            if not result[0].empty:
                identity_path = result[0].iloc[0]["identity"]
                identified_person = os.path.basename(os.path.dirname(identity_path))
            else:
                identified_person = "Unknown"
    except Exception:
        with lock:
            identified_person = "Unknown"
    finally:
        with lock:
            check_in_progress = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo_model.names[cls] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_roi = frame[y1:y2, x1:x2]

                
                if counter % frame_skip == 0:
                    with lock:
                        if not check_in_progress:
                            check_in_progress = True
                            threading.Thread(target=recognize_face, args=(face_roi.copy(),), daemon=True).start()

                
                with lock:
                    name = identified_person
                label = f"{name} {conf:.2f}" if name != "Unknown" else "Unknown"
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    counter += 1
    cv2.imshow("Person Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
