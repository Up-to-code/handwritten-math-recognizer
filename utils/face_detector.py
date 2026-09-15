import cv2
import numpy as np
import os
import pickle
from datetime import datetime


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.known_faces = {}
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_trained = False
        self._load_known_faces()

    def _load_known_faces(self):
        labels_file = os.path.join("data", "faces", "labels.pkl")
        if os.path.exists(labels_file):
            with open(labels_file, "rb") as f:
                data = pickle.load(f)
                self.known_faces = data.get("names", {})
                if data.get("encodings") is not None and len(data["encodings"]) > 0:
                    self.recognizer.read(
                        os.path.join("data", "faces", "recognizer.yml")
                    )
                    self.model_trained = True

    def _save_known_faces(self):
        labels_file = os.path.join("data", "faces", "labels.pkl")
        with open(labels_file, "wb") as f:
            pickle.dump(
                {
                    "names": self.known_faces,
                    "encodings": list(self.known_faces.values()),
                },
                f,
            )
        if self.model_trained:
            self.recognizer.write(os.path.join("data", "faces", "recognizer.yml"))

    def register_face(self, frame, student_id, student_name):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) == 0:
            return False, "No face detected"

        if len(faces) > 1:
            return False, "Multiple faces detected - show one face only"

        x, y, w, h = faces[0]
        face_img = gray[y : y + h, x : x + w]
        face_img = cv2.resize(face_img, (200, 200))

        student_dir = os.path.join("data", "faces", student_id)
        os.makedirs(student_dir, exist_ok=True)

        count = len(
            [
                f
                for f in os.listdir(student_dir)
                if f.endswith(".png")
            ]
        )
        cv2.imwrite(os.path.join(student_dir, f"{count + 1}.png"), face_img)

        self.known_faces[student_id] = student_name
        self._retrain_model()

        return True, f"Registered {student_name} with {count + 1} samples"

    def _retrain_model(self):
        faces = []
        labels = []
        label_map = {}
        label_id = 0

        for student_id in self.known_faces:
            student_dir = os.path.join("data", "faces", student_id)
            if not os.path.exists(student_dir):
                continue
            label_map[label_id] = student_id
            for img_file in os.listdir(student_dir):
                if img_file.endswith(".png"):
                    img = cv2.imread(
                        os.path.join(student_dir, img_file), cv2.IMREAD_GRAYSCALE
                    )
                    if img is not None:
                        img = cv2.resize(img, (200, 200))
                        faces.append(img)
                        labels.append(label_id)
            label_id += 1

        if len(faces) >= 2:
            self.recognizer.train(faces, np.array(labels))
            self.model_trained = True
            self.label_map = label_map
            self._save_known_faces()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        results = []
        for x, y, w, h in faces:
            face_img = gray[y : y + h, x : x + w]
            face_img = cv2.resize(face_img, (200, 200))

            student_id = "Unknown"
            confidence = 0

            if self.model_trained:
                label, conf = self.recognizer.predict(face_img)
                if conf < 70:
                    student_id = self.label_map.get(label, "Unknown")
                    confidence = 100 - conf

            results.append(
                {
                    "student_id": student_id,
                    "student_name": self.known_faces.get(student_id, "Unknown"),
                    "confidence": confidence,
                    "bbox": (x, y, w, h),
                }
            )

        return results

    def capture_face_for_training(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        annotated = frame.copy()
        for x, y, w, h in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return annotated, faces
