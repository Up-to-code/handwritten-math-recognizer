import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import re

CANVAS_W, CANVAS_H = 480, 480
PREVIEW_W = 200
NUM_CLASSES = 12
OP_PLUS, OP_MINUS = 10, 11


class NumberNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


def label_to_token(label):
    if 0 <= label <= 9:
        return str(label)
    if label == OP_PLUS:
        return "+"
    if label == OP_MINUS:
        return "-"
    return "?"


def eval_simple(expr):
    m = re.fullmatch(r"\s*(-?\d+)\s*([+-])\s*(-?\d+)\s*", expr)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return a + b if op == "+" else a - b


class DrawPad:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NumberNet().to(self.device)
        path = os.path.join("models", "expr_model.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.model.eval()
        else:
            print("No model found. Run: python3 retrain_robust.py")
            exit(1)

        self.canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
        self.drawing = False
        self.last = None
        self.number = None
        self.confidence = 0.0
        self.digits = []
        self.confirmed = None
        self.result = None
        self.window = "Draw a Number"

    def mouse(self, event, x, y, fl, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.canvas = cv2.line(self.canvas, self.last, (x, y), (0, 0, 0), 16,
                                   cv2.LINE_AA)
            self.last = (x, y)
            self.live_predict()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.live_predict()

    def live_predict(self):
        self.digits = []
        self.number = None
        self.confidence = 0.0

        rois = self.segment_digit_rois()
        if not rois:
            return

        digits = []
        for zone in rois:
            tensor = self.normalize_digit(zone["roi"])
            label, conf = self.predict_tensor(tensor)
            digits.append({"label": label, "token": label_to_token(label),
                           "conf": conf, "bbox": zone["bbox"]})

        self.digits = digits
        self.number = "".join(d["token"] for d in digits)
        self.confidence = min(d["conf"] for d in digits)
        self.result = eval_simple(self.number)

    def segment_digit_rois(self):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        total_area = sum(cv2.contourArea(c) for c in contours)
        big = [c for c in contours
               if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] > total_area * 0.015]

        mask = np.zeros(thresh.shape, dtype=np.uint8)
        for c in big:
            cv2.drawContours(mask, [c], -1, 255, -1)

        ys = np.nonzero(mask.sum(axis=1))[0]
        if len(ys) == 0:
            return []
        x_top, x_bot = ys.min(), ys.max()
        height = x_bot - x_top + 1

        xs = np.nonzero(mask.sum(axis=0))[0]
        if len(xs) == 0:
            return []

        zones = []
        start = xs[0]
        prev = xs[0]
        for v in xs[1:]:
            if v - prev > max(6, int(height * 0.08)):
                zones.append((start, prev))
                start = v
            prev = v
        zones.append((start, prev))

        rois = []
        for (z0, z1) in zones:
            w = z1 - z0 + 1
            if w < 8:
                continue
            roi = mask[x_top:x_bot + 1, z0:z1 + 1]
            rois.append({"roi": roi, "bbox": (z0, x_top, w, height)})
        return rois

    def normalize_digit(self, roi):
        h, w = roi.shape
        pad = max(w, h) // 4
        roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (28, 28))
        tensor = torch.FloatTensor(roi).unsqueeze(0) / 255.0
        return (tensor - 0.1307) / 0.3081

    def predict_tensor(self, tensor):
        with torch.no_grad():
            out = self.model(tensor.to(self.device).unsqueeze(0))
            probs = torch.softmax(out, 1)
            conf, pred = probs.max(1)
        return pred.item(), conf.item()

    def draw_ui(self):
        frame = np.full((CANVAS_H + 80, CANVAS_W + PREVIEW_W + 40, 3), (25, 25, 25), dtype=np.uint8)
        frame[:CANVAS_H, :CANVAS_W] = self.canvas
        cv2.rectangle(frame, (0, 0), (CANVAS_W, CANVAS_H), (200, 200, 200), 2)

        for d in self.digits:
            x, y, w, h = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(d["token"]), (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        px = CANVAS_W + 20
        cv2.putText(frame, "DETECTED", (px, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if self.number is not None:
            size = 4.0 if len(self.number) <= 2 else (3.0 if len(self.number) <= 3 else 2.2)
            cv2.putText(frame, self.number, (px + 10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 7)
            per = "  ".join(f"{d['token']}:{d['conf']:.0%}" for d in self.digits)
            cv2.putText(frame, per, (px, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 120), 1)
            if self.number.replace("+", "").replace("-", "").isdigit():
                places = {1: "ones", 2: "tens", 3: "hundreds", 4: "thousands"}.get(len(self.number), "")
                cv2.putText(frame, f"{len(self.number)} digit(s) ({places})", (px, 258),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 180, 100), 1)
            if self.result is not None:
                cv2.putText(frame, f"= {self.result}", (px, 258),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 4)
            cv2.putText(frame, f"overall: {self.confidence:.0%}", (px, 276),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 120), 1)
        else:
            cv2.putText(frame, "draw a number", (px + 10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
            cv2.putText(frame, "or an expression", (px + 10, 178),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            cv2.putText(frame, "like 12+5 or 9-4", (px + 10, 202),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        if self.confirmed is not None:
            cv2.putText(frame, f"FINAL: {self.confirmed[0]} ({self.confirmed[1]:.0%})", (px, 296),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        bar_y = CANVAS_H + 15
        cv2.putText(frame, "DRAW with mouse | c = clear | Enter = confirm | q = quit",
                    (15, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if not self.drawing and self.number is not None and self.confidence < 0.6:
            cv2.putText(frame, "LOW CONFIDENCE - draw digits bigger with gaps", (15, CANVAS_H - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame

    def run(self):
        print("=" * 60)
        print("   DRAW PAD - draw numbers AND simple + / - math")
        print("=" * 60)
        print("  Draw with the mouse | c = clear | Enter = confirm | q = quit")
        print("  Examples: 7   35   12+5   100-40")
        print("  Tip: leave a small gap between each symbol")
        print("=" * 60)

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse)

        while True:
            frame = self.draw_ui()
            cv2.imshow(self.window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key == ord("c"):
                self.canvas[:] = 255
                self.number = None
                self.confidence = 0.0
                self.digits = []
                self.confirmed = None
                self.result = None
            elif key == 13:
                if self.number is not None:
                    self.confirmed = (self.number, self.confidence)
                    if self.result is not None:
                        print(f"CONFIRMED: {self.number} = {self.result}  (confidence {self.confidence:.0%})")
                    else:
                        print(f"CONFIRMED: {self.number}  (confidence {self.confidence:.0%})")
                else:
                    print("Nothing drawn yet.")

        cv2.destroyAllWindows()
        print("Closed.")


if __name__ == "__main__":
    app = DrawPad()
    app.run()