import cv2
import numpy as np


def detect_and_compute_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def draw_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,
     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches


def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def recognize_face(live_face, kp_ref, des_ref):
    kp_live, des_live = detect_and_compute_features(live_face)
    if des_ref is None or des_live is None:
        return "Unknown"

    matches = match_features(des_ref, des_live)
    if len(matches) > 10:  # Adjust this threshold based on experimentation
        return "Bill Gates"
    else:
        return "Unknown"


# Load the reference image (Bill Gates photo)
ref_image = cv2.imread('Bill.jpg')
if ref_image is None:
    print("Error: Reference image not found.")
    exit()

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect and compute features for the reference image
kp_ref, des_ref = detect_and_compute_features(ref_image)

# Capture a live image (use a webcam or any camera input)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

print("Press 'q' to capture a live photo.")
while True:
    ret, live_frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    faces = detect_faces(live_frame, face_cascade)

    for (x, y, w, h) in faces:
        live_face = live_frame[y:y + h, x:x + w]
        label = recognize_face(live_face, kp_ref, des_ref)
        color = (0, 255, 0) if label == "Bill Gates" else (0, 0, 255)
        cv2.rectangle(live_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(live_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Live Feed', live_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



