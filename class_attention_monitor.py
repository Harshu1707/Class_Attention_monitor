import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8s.pt")   # CHANGE TO yolov8s FOR BETTER PHONE DETECTION

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10)

# EAR Calculation
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])

    vert1 = np.linalg.norm(p2 - p4)
    vert2 = np.linalg.norm(p3 - p1)
    horiz = np.linalg.norm(p5 - p6)

    return (vert1 + vert2) / (2.0 * horiz)

# Eye Indexes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

EAR_THRESHOLD = 0.20

cap = cv2.VideoCapture(0)

# IOU check
def box_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        return True
    return False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = model(frame, conf=0.5)

    person_boxes = []
    phone_boxes = []

    # STEP 1: Extract persons + phones
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:   # person
                person_boxes.append((x1, y1, x2, y2))
            elif cls == 67:  # phone
                phone_boxes.append((x1, y1, x2, y2))

    # STEP 2: Process each person
    for (x1, y1, x2, y2) in person_boxes:

        # Check phone usage FIRST
        phone_usage = False
        for ph in phone_boxes:
            if box_overlap((x1, y1, x2, y2), ph):
                phone_usage = True

        # Crop person face area for FaceMesh
        person_img = frame[y1:y2, x1:x2]
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_img)

        attentive = True
        direction = "Forward"
        eye_status = "Eyes Open"

        if output.multi_face_landmarks:
            for landmarks in output.multi_face_landmarks:
                lm = landmarks.landmark

                left_ear = eye_aspect_ratio(lm, LEFT_EYE)
                right_ear = eye_aspect_ratio(lm, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2

                # Eyes closed
                if avg_ear < EAR_THRESHOLD:
                    eye_status = "Eyes Closed"
                    attentive = False

                # Head direction
                nose = lm[1].x
                if nose < 0.40:
                    direction = "Looking Right"
                    attentive = False
                elif nose > 0.60:
                    direction = "Looking Left"
                    attentive = False

        # FINAL DECISION
        if phone_usage:
            status_text = "NOT ATTENTIVE | Using Phone"
            attentive = False
        else:
            status_text = "ATTENTIVE" if attentive else "NOT ATTENTIVE"

        color = (0, 255, 0) if attentive else (0, 0, 255)

        # Draw Person Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{status_text} | {direction} | {eye_status}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw phone boxes
    for (px1, py1, px2, py2) in phone_boxes:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(frame, "Phone", (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Attention Monitor + Phone Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
