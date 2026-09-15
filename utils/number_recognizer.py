import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import cv2
import os


class NumberNet(nn.Module):
    def __init__(self):
        super(NumberNet, self).__init__()
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
        x = self.fc2(x)
        return x


class NumberRecognizer:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NumberNet().to(self.device)
        self.model_path = model_path or os.path.join("models", "number_model.pth")
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()

    def train_on_mnist(self, epochs=5, batch_size=64, lr=0.001):
        train_dataset = MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        test_dataset = MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

            test_loss = 0
            correct = 0
            self.model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()

            accuracy = 100.0 * correct / len(test_dataset)
            print(
                f"Epoch {epoch + 1}: Test Accuracy: {accuracy:.2f}%, Loss: {test_loss / len(test_loader):.4f}"
            )
            self.model.train()

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def preprocess_roi(self, roi):
        if roi is None or roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            digit = thresh[y : y + h, x : x + w]
            pad = max(w, h) // 4
            digit = cv2.copyMakeBorder(
                digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
            )
        else:
            digit = thresh

        digit = cv2.resize(digit, (28, 28))
        return digit

    def predict(self, roi):
        digit = self.preprocess_roi(roi)
        if digit is None:
            return -1, 0.0

        tensor = self.transform(digit).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)

        return predicted.item(), confidence.item()

    def predict_from_canvas(self, canvas):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if len(canvas.shape) == 3 else canvas

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return -1, 0.0, None

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit_roi = gray[y : y + h, x : x + w]
        pad = max(w, h) // 4
        digit_roi = cv2.copyMakeBorder(
            digit_roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )

        result, confidence = self.predict(
            cv2.cvtColor(digit_roi, cv2.COLOR_GRAY2BGR)
            if len(digit_roi.shape) == 2
            else digit_roi
        )

        return result, confidence, (x, y, w, h)
