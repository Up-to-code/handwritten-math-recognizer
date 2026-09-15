import cv2
import numpy as np
import platform
import os
import sys
import time


class SmartCamera:
    def __init__(self):
        self.window = "Smart Classroom Camera"
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.source_label = "NO CAMERA"
        self.fps = 0
        self.face_hits = {}
        self.preview = None

    def check_macos_permission(self):
        if platform.system() == "Darwin":
            print("""
[macOS] To use the computer webcam you must allow camera access:
  System Settings > Privacy & Security > Camera > enable your terminal/app
If you cannot enable it, use your PHONE camera over the network instead (option 2).
""")

    def open_source(self):
        print("\n=== CHOOSE CAMERA SOURCE ===")
        print("  1: Computer webcam")
        print("  2: Phone camera over network (needs IP Webcam app)")
        print("  3: Load a folder of test images")
        choice = input("> ").strip()

        if choice == "2":
            print("""
[PHONE SETUP - 30 seconds]
  1. Install "IP Webcam" app from the phone app store
  2. Open it and tap the big START SERVER button at the bottom
  3. The app shows a URL like:  http://192.168.1.20:8080/video
""")
            url = input("Enter the phone URL shown in IP Webcam  (e.g. http://192.168.1.20:8080/video)\n> ").strip()
            if not url:
                url = "http://192.168.1.10:8080/video"
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print("Could not connect. Check: phone and computer on same Wi-Fi, server started.")
                return None, url
            return cap, ("PHONE " + url)
        elif choice == "3":
            folder = input("Folder with images: ").strip() or "data/test_images"
            if not os.path.exists(folder):
                print(f"Folder not found: {folder}")
                return None, folder
            self.preview = {"folder": folder, "files": sorted(
                [f for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".jpeg"))]
            ), "idx": 0}
            return "IMAGES", folder
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Webcam not available or permission denied.")
                return None, "webcam"
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap, "WEBCAM"

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))

        now = time.time()
        for (x, y, w, h) in faces:
            key = (x // 60, y // 60)
            self.face_hits[key] = now
        self.face_hits = {k: v for k, v in self.face_hits.items() if now - v < 1.5}
        stability = min(1.0, len(self.face_hits) / 20.0)

        return faces, len(faces), stability

    def draw_overlay(self, frame, num_faces, stability):
        h, w = frame.shape[:2]
        cv2.line(frame, (0, 36), (w, 36), (70, 70, 70), 1)

        cv2.putText(frame, "SMART CAMERA", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Source: {self.source_label[:45]}", (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"{self.fps:.0f} FPS | Faces: {num_faces} | Score: {stability:.0%}",
                    (w - 320, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 120), 1)

    def msleep_consistent(self, frame_time, target_fps=15):
        elapsed = time.time() - frame_time
        if elapsed < 1.0 / target_fps:
            time.sleep(1.0 / target_fps - elapsed)

    def run(self):
        print("=" * 60)
        print("   SMART CLASSROOM - REAL-TIME CAMERA")
        print("=" * 60)
        self.check_macos_permission()

        cap, name = self.open_source()
        if self.preview is None and cap is None:
            print("No camera source available. Exiting.")
            return
        self.source_label = name

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        last = time.time()

        while True:
            frame_time = time.time()

            if self.preview is not None:
                files = self.preview["files"]
                frame = None
                while files and frame is None:
                    path = os.path.join(self.preview["folder"], files[self.preview["idx"]])
                    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        frame = cv2.resize(frame, (640, 480))
                    self.preview["idx"] = (self.preview["idx"] + 1) % len(files)
                    time.sleep(1)
                if frame is None:
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Lost camera feed. Exiting.")
                    break

            faces, num_faces, stability = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, f"Face {stability:.0%}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.fps = 1.0 / max(0.001, time.time() - last)
            last = time.time()
            self.draw_overlay(frame, num_faces, stability)

            cv2.imshow(self.window, frame)
            self.msleep_consistent(frame_time)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

        if cap is not None and cap != "IMAGES":
            cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    app = SmartCamera()
    app.run()