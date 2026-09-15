import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class NumberNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class SmartClassroom:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NumberNet().to(self.device)
        self.model_path = os.path.join("models", "number_model.pth")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.stats = {
            "total_predictions": 0,
            "correct": 0,
            "incorrect": 0,
            "test_results": [],
        }
        self.profiles = {}
        self.load_model()
        self.load_stats()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
        else:
            print("No saved model found. Train first with 't' key.")

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_stats(self):
        if os.path.exists("stats.json"):
            with open("stats.json", "r") as f:
                self.stats = json.load(f)

    def save_stats(self):
        with open("stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

    def train_model(self, epochs=3):
        from torchvision.datasets import MNIST
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader

        print("Downloading MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_set = MNIST(root="./data", train=True, download=True, transform=transform)
        test_set = MNIST(root="./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            self.model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()

            acc = 100.0 * correct / len(test_set)
            print(f"  Epoch {epoch+1}: Accuracy = {acc:.1f}%")
            self.model.train()

        self.save_model()
        print("Training complete!")

    def predict_digit(self, img):
        if img is None:
            return -1, 0.0

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            digit = thresh[y:y+h, x:x+w]
            pad = max(w, h) // 4
            digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        else:
            digit = thresh

        digit = cv2.resize(digit, (28, 28))
        tensor = torch.FloatTensor(digit).unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.1307) / 0.3081
        tensor = tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)

        return predicted.item(), confidence.item()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        return faces

    def draw_dashboard(self, frame):
        h, w = frame.shape[:2]
        panel = np.zeros((h, 260, 3), dtype=np.uint8)
        panel[:] = (35, 35, 35)

        y = 30
        cv2.putText(panel, "SMART CLASSROOM", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        y += 15
        cv2.line(panel, (15, y), (245, y), (80, 80, 80), 1)
        y += 25

        cv2.putText(panel, "STATS", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 25
        cv2.putText(panel, f"Predictions: {self.stats['total_predictions']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22
        cv2.putText(panel, f"Correct: {self.stats['correct']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Incorrect: {self.stats['incorrect']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        y += 22

        total = self.stats["total_predictions"]
        if total > 0:
            pct = self.stats["correct"] / total * 100
            bar_w = int(pct * 2.0)
            cv2.rectangle(panel, (15, y), (15 + bar_w, y + 15), (0, 180, 0), -1)
            cv2.rectangle(panel, (15, y), (215, y + 15), (100, 100, 100), 1)
            cv2.putText(panel, f"{pct:.0f}%", (220, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 35

        cv2.line(panel, (15, y), (245, y), (80, 80, 80), 1)
        y += 20

        cv2.putText(panel, "CONTROLS", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 22
        controls = [
            ("T", "Train Model"),
            ("G", "Generate Tests"),
            ("P", "Run Test Images"),
            ("C", "Toggle Camera"),
            ("Q", "Quit"),
        ]
        for key, desc in controls:
            cv2.putText(panel, f"[{key}] {desc}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += 20

        y += 10
        cv2.line(panel, (15, y), (245, y), (80, 80, 80), 1)
        y += 20

        cv2.putText(panel, "RECENT RESULTS", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 25

        recent = self.stats["test_results"][-5:]
        for r in reversed(recent):
            color = (0, 255, 0) if r["correct"] else (0, 0, 255)
            status = "OK" if r["correct"] else "MISS"
            cv2.putText(panel, f"{r['filename']}: {r['predicted']}={r['actual']} [{status}]",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y += 18

        combined = np.hstack([frame, panel])
        return combined

    def run_test_images(self):
        test_dir = "data/test_images"
        if not os.path.exists(test_dir):
            print("No test images. Press 'G' to generate first.")
            return None

        results = []
        images = []

        for f in sorted(os.listdir(test_dir)):
            if f.startswith("test_") and f.endswith(".png"):
                img_path = os.path.join(test_dir, f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                parts = f.replace(".png", "").split("_")
                actual = int(parts[1])
                style = parts[2]

                predicted, confidence = self.predict_digit(img)
                is_correct = predicted == actual

                self.stats["total_predictions"] += 1
                if is_correct:
                    self.stats["correct"] += 1
                else:
                    self.stats["incorrect"] += 1

                result = {
                    "filename": f,
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": round(confidence, 3),
                    "correct": is_correct,
                    "style": style,
                    "timestamp": datetime.now().isoformat(),
                }
                self.stats["test_results"].append(result)
                results.append(result)

                color = (0, 255, 0) if is_correct else (0, 0, 255)
                status = "CORRECT" if is_correct else "WRONG"
                print(f"  {f}: Predicted={predicted}, Actual={actual}, Conf={confidence:.2%} -> {status}")

                img_display = cv2.resize(img, (200, 200))
                img_color = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
                cv2.putText(img_color, f"P:{predicted} A:{actual}", (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(img_color, f"{confidence:.0%}", (5, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.rectangle(img_color, (0, 0), (199, 199), color, 2)
                images.append(img_color)

        self.save_stats()

        if images:
            cols = min(4, len(images))
            rows = (len(images) + cols - 1) // cols

            preview = np.zeros((rows * 220 + 40, cols * 220 + 20, 3), dtype=np.uint8)
            preview[:] = (30, 30, 30)

            cv2.putText(preview, "TEST RESULTS", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                y = row * 220 + 40
                x = col * 220 + 10
                preview[y:y+200, x:x+200] = img

            correct = sum(1 for r in results if r["correct"])
            total = len(results)
            cv2.putText(preview, f"Score: {correct}/{total} ({correct/total*100:.0f}%)", (15, rows * 220 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return preview

        return None

    def generate_test_images(self):
        from generate_tests import generate_test_images
        generated, preview = generate_test_images()
        print(f"Generated {len(generated)} test images")
        return generated

    def run(self):
        print("=" * 55)
        print("     SMART CLASSROOM - Face & Number Detection")
        print("=" * 55)
        print("  [T] Train Model      [G] Generate Test Images")
        print("  [P] Run Tests        [C] Toggle Camera")
        print("  [Q] Quit")
        print("=" * 55)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        camera_on = cap.isOpened()
        test_preview = None

        while True:
            if camera_on:
                ret, frame = cap.read()
                if not ret:
                    camera_on = False
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    frame[:] = (30, 30, 30)
                else:
                    frame = cv2.flip(frame, 1)
                    faces = self.detect_faces(frame)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (30, 30, 30)
                cv2.putText(frame, "Camera OFF - Press [C] to enable", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

            display = self.draw_dashboard(frame)
            cv2.imshow("Smart Classroom", display)

            if test_preview is not None:
                cv2.imshow("Test Results", test_preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("t"):
                print("\nTraining model on MNIST...")
                self.train_model(epochs=3)

            elif key == ord("g"):
                print("\nGenerating test images...")
                self.generate_test_images()

            elif key == ord("p"):
                print("\nRunning test images...")
                test_preview = self.run_test_images()

            elif key == ord("c"):
                if not camera_on:
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(0)
                    camera_on = cap.isOpened()
                else:
                    camera_on = False

        cap.release()
        cv2.destroyAllWindows()
        self.save_stats()
        print("Done!")


if __name__ == "__main__":
    app = SmartClassroom()
    app.run()
