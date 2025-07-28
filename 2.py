import threading
import cv2
from deepface import DeepFace
from ultralytics import YOLO

face_reference_img = cv2.imread("reference2.jpg")  
yolo_model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


face_match = False
check_in_progress = False
counter = 0
frame_skip = 30
lock = threading.Lock()

def check_face(frame):
    global face_match, check_in_progress
    try:
        result = DeepFace.verify(frame, face_reference_img, enforce_detection=False)
        with lock:
            face_match = result['verified']
    except Exception:
        with lock:
            face_match = False
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if counter % frame_skip == 0:
        with lock:
            if not check_in_progress:
                check_in_progress = True
                threading.Thread(target=check_face, args=(frame.copy(),), daemon=True).start()
    counter += 1

    with lock:
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)


    cv2.imshow("Human Detection + Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
