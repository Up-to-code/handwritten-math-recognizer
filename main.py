import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.face_detector import FaceDetector
from utils.number_recognizer import NumberRecognizer
from utils.profile_manager import StudentProfileManager


class SmartClassroom:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.number_recognizer = NumberRecognizer()
        self.profile_manager = StudentProfileManager()

        self.cap = None
        self.drawing = False
        self.drawing_mode = False
        self.canvas = None
        self.last_x, self.last_y = None, None

        self.state = "MAIN_MENU"
        self.selected_student = None
        self.current_question = None
        self.question_number = 0
        self.show_answer = False

        self.colors = {
            "bg": (30, 30, 30),
            "text": (255, 255, 255),
            "success": (0, 255, 0),
            "error": (0, 0, 255),
            "info": (255, 200, 0),
            "drawing": (0, 200, 255),
        }

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.cap.isOpened()

    def draw_button(self, frame, text, x, y, w, h, color=(100, 100, 100)):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = x + (w - text_size[0]) // 2
        ty = y + (h + text_size[1]) // 2
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return (x, y, w, h)

    def is_button_clicked(self, x, y, btn):
        bx, by, bw, bh = btn
        return bx <= x <= bx + bw and by <= y <= by + bh

    def main_menu(self, frame):
        h, w = frame.shape[:2]

        cv2.putText(frame, "SMART CLASSROOM", (w // 2 - 180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors["info"], 2)

        buttons = {
            "train": self.draw_button(frame, "Train Number Model", w // 2 - 120, 100, 240, 40, (80, 120, 80)),
            "register": self.draw_button(frame, "Register Student", w // 2 - 120, 160, 240, 40, (80, 80, 120)),
            "quiz": self.draw_button(frame, "Start Quiz Mode", w // 2 - 120, 220, 240, 40, (120, 80, 80)),
            "leaderboard": self.draw_button(frame, "Leaderboard", w // 2 - 120, 280, 240, 40, (100, 100, 80)),
            "quit": self.draw_button(frame, "Quit", w // 2 - 120, 340, 240, 40, (60, 60, 120)),
        }

        cv2.putText(frame, "Press 'r' to Register | 'q' to Quiz | 't' to Train", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors["text"], 1)

        return buttons

    def register_mode(self, frame):
        h, w = frame.shape[:2]

        cv2.putText(frame, "REGISTER STUDENT", (w // 2 - 180, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors["info"], 2)

        annotated, faces = self.face_detector.capture_face_for_training(frame)

        for x, y, fw, fh in faces:
            cv2.putText(annotated, "Face Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["success"], 2)

        info_y = h - 80
        cv2.putText(annotated, "Enter student ID in terminal, then press ENTER in OpenCV", (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
        cv2.putText(annotated, "Press ESC to go back", (20, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["info"], 1)

        if self.register_data:
            cv2.putText(annotated, f"Registering: {self.register_data['name']}", (20, info_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["success"], 1)

        return annotated

    def quiz_mode(self, frame):
        h, w = frame.shape[:2]

        detected_faces = self.face_detector.detect_faces(frame)

        for face in detected_faces:
            x, y, fw, fh = face["bbox"]
            color = self.colors["success"] if face["student_id"] != "Unknown" else self.colors["error"]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            label = f"{face['student_name']} ({face['confidence']:.0f}%)"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        panel_x = w - 250
        cv2.rectangle(frame, (panel_x, 0), (w, h), (40, 40, 40), -1)

        cv2.putText(frame, "QUIZ MODE", (panel_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["info"], 2)

        cv2.putText(frame, f"Q#: {self.question_number}", (panel_x + 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)

        if self.current_question is not None:
            cv2.putText(frame, f"Correct: {self.current_question}", (panel_x + 10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["success"], 1)

        if detected_faces:
            self.selected_student = detected_faces[0]
            name = self.selected_student["student_name"]
            cv2.putText(frame, f"Student: {name}", (panel_x + 10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
        else:
            cv2.putText(frame, "No face detected", (panel_x + 10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["error"], 1)

        btn_draw = self.draw_button(frame, "Drawing Pad", panel_x + 10, 150, 220, 35, (80, 120, 80))
        btn_answer = self.draw_button(frame, "Submit Answer", panel_x + 10, 200, 220, 35, (120, 80, 80))
        btn_next = self.draw_button(frame, "Next Question", panel_x + 10, 250, 220, 35, (80, 80, 120))

        buttons = {"draw": btn_draw, "answer": btn_answer, "next": btn_next}

        if self.drawing_mode:
            self.draw_overlay(frame)

        return frame, buttons

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (10, 10), (320, 250), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (320, 250), self.colors["drawing"], 2)

        cv2.putText(overlay, "DRAW NUMBER (0-9)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["drawing"], 1)

        if self.canvas is not None:
            mask = self.canvas > 0
            if mask.any():
                roi = overlay[12:248, 12:318]
                canvas_resized = cv2.resize(self.canvas, (306, 236))
                mask_resized = canvas_resized > 0
                roi[mask_resized] = canvas_resized[mask_resized]

        cv2.putText(overlay, "c: Clear | Enter: Submit | ESC: Cancel", (15, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors["text"], 1)

        frame[:] = overlay

    def handle_mouse(self, event, x, y, flags, param):
        if not self.drawing_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_x, self.last_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.canvas is None:
                self.canvas = np.zeros((236, 306), dtype=np.uint8)

            cx, cy = x - 12, y - 12
            lx, ly = self.last_x - 12, self.last_y - 12

            if 0 <= cx < 306 and 0 <= cy < 236:
                cv2.line(self.canvas, (lx, ly), (cx, cy), 255, 8)

            self.last_x, self.last_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def run(self):
        if not self.start_camera():
            print("ERROR: Cannot open camera")
            return

        cv2.namedWindow("Smart Classroom")
        cv2.setMouseCallback("Smart Classroom", self.handle_mouse)

        print("=" * 50)
        print("    SMART CLASSROOM - Face & Number Detection")
        print("=" * 50)
        print("Controls:")
        print("  'r' - Register Student")
        print("  'q' - Quiz Mode")
        print("  't' - Train Number Model")
        print("  'l' - Leaderboard")
        print("  ESC - Back/Quit")
        print("=" * 50)

        self.register_data = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if self.state == "MAIN_MENU":
                buttons = self.main_menu(frame)

            elif self.state == "REGISTER":
                frame = self.register_mode(frame)
                buttons = {}

            elif self.state == "QUIZ":
                frame, buttons = self.quiz_mode(frame)

            elif self.state == "LEADERBOARD":
                self.draw_leaderboard(frame)
                buttons = {}

            else:
                buttons = {}

            cv2.imshow("Smart Classroom", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                if self.state == "DRAWING":
                    self.drawing_mode = False
                    self.canvas = None
                    self.state = "QUIZ"
                elif self.state != "MAIN_MENU":
                    self.state = "MAIN_MENU"
                    self.drawing_mode = False
                    self.canvas = None
                else:
                    break

            elif key == ord("r") and self.state == "MAIN_MENU":
                print("\n--- Register Student ---")
                student_id = input("Enter Student ID: ").strip()
                student_name = input("Enter Student Name: ").strip()
                if student_id and student_name:
                    self.register_data = {"id": student_id, "name": student_name}
                    self.state = "REGISTER"
                    print("Show face to camera. Press ENTER in OpenCV window when ready...")

            elif key == 13 and self.state == "REGISTER":
                if self.register_data:
                    ret, msg = self.face_detector.register_face(
                        frame, self.register_data["id"], self.register_data["name"]
                    )
                    if ret:
                        profile = self.profile_manager.get_profile(self.register_data["id"])
                        if profile is None:
                            self.profile_manager.create_profile(
                                self.register_data["id"], self.register_data["name"]
                            )
                        print(f"SUCCESS: {msg}")
                    else:
                        print(f"ERROR: {msg}")
                    self.register_data = None
                    self.state = "MAIN_MENU"

            elif key == ord("q") and self.state == "MAIN_MENU":
                self.state = "QUIZ"
                self.question_number += 1
                self.current_question = np.random.randint(0, 10)
                print(f"\nQuestion #{self.question_number}: Show number {self.current_question}")

            elif key == ord("t") and self.state == "MAIN_MENU":
                print("\nTraining number recognition model on MNIST...")
                print("This may take a few minutes...")
                self.number_recognizer.train_on_mnist(epochs=5)
                print("Training complete!")

            elif key == ord("l") and self.state == "MAIN_MENU":
                self.state = "LEADERBOARD"

            elif key == ord("c") and self.state == "QUIZ":
                self.drawing_mode = not self.drawing_mode
                if self.drawing_mode:
                    self.canvas = np.zeros((236, 306), dtype=np.uint8)
                    self.state_prev = "QUIZ"

            elif key == 13 and self.state == "QUIZ" and self.drawing_mode:
                if self.canvas is not None:
                    result, confidence, _ = self.number_recognizer.predict_from_canvas(self.canvas)
                    if result >= 0:
                        print(f"Detected number: {result} (confidence: {confidence:.2%})")

                        if self.selected_student and self.current_question is not None:
                            student_id = self.selected_student["student_id"]
                            if student_id != "Unknown":
                                outcome = self.profile_manager.record_answer(
                                    student_id,
                                    self.question_number,
                                    self.current_question,
                                    result,
                                )
                                if outcome:
                                    status = "CORRECT" if outcome["is_correct"] else "WRONG"
                                    print(f"  {status}! Score: {outcome['percentage']:.1f}% | Grade: {outcome['profile']['grade']}")
                            else:
                                print("  Student not recognized - register first!")

                    self.drawing_mode = False
                    self.canvas = None

            elif key == 13 and self.state == "QUIZ" and not self.drawing_mode:
                pass

            elif self.state == "MAIN_MENU":
                pass

        self.cap.release()
        cv2.destroyAllWindows()

    def draw_leaderboard(self, frame):
        h, w = frame.shape[:2]
        frame[:] = (30, 30, 30)

        cv2.putText(frame, "LEADERBOARD", (w // 2 - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors["info"], 2)

        leaderboard = self.profile_manager.get_leaderboard()

        y = 80
        for i, entry in enumerate(leaderboard[:10]):
            color = self.colors["success"] if entry["percentage"] >= 70 else self.colors["text"]
            text = f"{i + 1}. {entry['name']} - {entry['correct']}/{entry['total']} ({entry['percentage']}%) [{entry['grade']}]"
            cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y += 35

        if not leaderboard:
            cv2.putText(frame, "No students registered yet", (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["text"], 1)

        cv2.putText(frame, "Press ESC to go back", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["info"], 1)


if __name__ == "__main__":
    app = SmartClassroom()
    app.run()
