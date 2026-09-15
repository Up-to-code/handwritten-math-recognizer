import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import random
import os

NUM_CLASSES = 12
OP_PLUS, OP_MINUS = 10, 11
FONT_PATH = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"


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


def render_operator(op, target=28):
    canvas_size = 64
    img = Image.new("L", (canvas_size, canvas_size), 0)

    if op == "+":
        draw = ImageDraw.Draw(img)
        cx = cy = canvas_size // 2
        thickness = random.randint(3, 6)
        length = random.randint(36, 48)
        draw.line([(cx, cy - length // 2), (cx, cy + length // 2)], fill=255, width=thickness)
        draw.line([(cx - length // 2, cy), (cx + length // 2, cy)], fill=255, width=thickness)
        rot = random.uniform(-18, 18)
        img = img.rotate(rot, fillcolor=0)
    else:
        draw = ImageDraw.Draw(img)
        cy = canvas_size // 2 + random.randint(-4, 4)
        thickness = random.randint(3, 6)
        length = random.randint(30, 46)
        draw.line([(canvas_size // 2 - length // 2, cy),
                   (canvas_size // 2 + length // 2, cy)], fill=255, width=thickness)
        rot = random.uniform(-18, 18)
        img = img.rotate(rot, fillcolor=0)

    arr = np.array(img)
    ys, xs = np.nonzero(arr > 128)
    if len(ys) == 0:
        return torch.zeros(28, 28)
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    pad = int(max(w, h) * 0.25)
    crop = np.pad(crop, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    crop = cv2_resize(crop, target)

    if random.random() < 0.25:
        noise = np.random.normal(0, 20, crop.shape).astype(np.int16)
        crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    tensor = torch.FloatTensor(crop) / 255.0
    return (tensor - 0.1307) / 0.3081


def render_digit(d, target=28):
    canvas_size = 128
    img = Image.new("L", (canvas_size, canvas_size), 0)
    dr = ImageDraw.Draw(img)
    fsize = random.randint(96, 126)
    font = ImageFont.truetype(FONT_PATH, fsize)
    bbox = dr.textbbox((0, 0), str(d), font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0 = canvas_size // 2 - tw // 2 - bbox[0]
    y0 = canvas_size // 2 - th // 2 - bbox[1]
    dr.text((x0, y0), str(d), fill=255, font=font,
            stroke_width=random.choice([0, 1, 2]) if random.random() < 0.6 else 0,
            stroke_fill=255)
    if random.random() < 0.7:
        img = img.rotate(random.uniform(-15, 15), fillcolor=0, resample=Image.BICUBIC)
    if random.random() < 0.5:
        ox, oy = random.randint(-5, 5), random.randint(-5, 5)
        img = img.transform(img.size, Image.AFFINE, (1, 0, ox, 0, 1, oy),
                            resample=Image.BICUBIC, fillcolor=0)

    arr = np.array(img)
    ys, xs = np.nonzero(arr > 128)
    if len(ys) == 0 or len(xs) == 0:
        return torch.zeros(28, 28)
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    pad = int(max(w, h) * 0.25)
    crop = np.pad(crop, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    if random.random() < 0.5:
        k = random.choice([2, 3, 4])
        crop = cv2.dilate(crop, np.ones((k, k), np.uint8), iterations=1)
    scale = random.uniform(0.85, 1.15)
    crop = cv2.resize(crop, (max(2, int(target * scale)), max(2, int(target * scale))),
                      interpolation=cv2.INTER_AREA)
    crop = cv2_resize(crop, target)

    if random.random() < 0.25:
        noise = np.random.normal(0, 20, crop.shape).astype(np.int16)
        crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    tensor = torch.FloatTensor(crop) / 255.0
    return (tensor - 0.1307) / 0.3081


def cv2_resize(crop, target):
    import cv2
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)


class MixedDataset(Dataset):
    def __init__(self, mnist, n_ops, n_digits):
        self.data = mnist.data
        self.targets = mnist.targets
        self.index = np.arange(len(self.data))
        random.shuffle(self.index)
        self.digits = []
        for _ in range(n_digits):
            d = random.randint(0, 9)
            self.digits.append((render_digit(str(d)), d))
        random.shuffle(self.digits)
        self.ops = []
        counts = {"plus": 0, "minus": 0}
        while counts["plus"] < n_ops or counts["minus"] < n_ops:
            if counts["plus"] < n_ops:
                self.ops.append((render_operator("+"), OP_PLUS))
                counts["plus"] += 1
            if counts["minus"] < n_ops:
                self.ops.append((render_operator("-"), OP_MINUS))
                counts["minus"] += 1
        random.shuffle(self.ops)

    def __len__(self):
        return len(self.index) + len(self.digits) + len(self.ops)

    def __getitem__(self, idx):
        if idx < len(self.index):
            i = self.index[idx]
            img = self.data[i].numpy()
            label = int(self.targets[i])

            pil = Image.fromarray(img, mode="L")

            if random.random() < 0.5:
                pil = pil.rotate(random.uniform(-18, 18), fillcolor=0, resample=Image.BICUBIC)

            if random.random() < 0.4:
                pil = pil.transform(
                    pil.size, Image.AFFINE,
                    (1, 0, random.randint(-3, 3), 0, 1, random.randint(-3, 3)),
                    resample=Image.BICUBIC, fillcolor=0,
                )

            if label == 0 and random.random() < 0.35:
                arr = np.array(pil)
                ys, xs = np.nonzero(arr < 128)
                if len(ys):
                    xo = int(max(xs))
                    top = ys[np.where(xs == xo)]
                    dy = max(1, int(min(top)) * 0.4) if len(top) else 4
                    draw = ImageDraw.Draw(pil)
                    draw.line([(min(xo, 26), 27), (min(xo, 26), 27 - dy)], fill=0, width=2)

            tensor = transforms.ToTensor()(pil).squeeze(0)
            tensor = (tensor - 0.1307) / 0.3081
            return tensor, torch.tensor(label, dtype=torch.long)

        if idx < len(self.index) + len(self.digits):
            t, l = self.digits[idx - len(self.index)]
            return t, torch.tensor(l, dtype=torch.long)

        t, l = self.ops[idx - len(self.index) - len(self.digits)]
        return t, torch.tensor(l, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_full = MNIST(root="./data", train=True, download=False)
    test_set = MNIST(root="./data", train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,)),
                     ]))

    dataset = MixedDataset(train_full, n_ops=5000, n_digits=10000)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    OPERATOR_SAMPLES = 1500
    op_plus = [(render_operator("+", 28), OP_PLUS) for _ in range(OPERATOR_SAMPLES)]
    op_minus = [(render_operator("-", 28), OP_MINUS) for _ in range(OPERATOR_SAMPLES)]
    op_loader = DataLoader([
        (t.unsqueeze(0), torch.tensor(l, dtype=torch.long)) for t, l in (op_plus + op_minus)
    ], batch_size=256)

    RENDERED_SAMPLES = 500
    rendered = []
    for _ in range(10 * RENDERED_SAMPLES):
        d = random.randint(0, 9)
        rendered.append((render_digit(str(d)), d))
    font_loader = DataLoader([
        (t.unsqueeze(0), torch.tensor(l, dtype=torch.long)) for t, l in rendered
    ], batch_size=256)

    model = NumberNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 4
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data.unsqueeze(1))
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 150 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}, batch {batch_idx}, loss {loss.item():.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(1)
                correct += pred.eq(target).sum().item()
        print(f"  Epoch {epoch+1}: MNIST accuracy {100*correct/len(test_set):.2f}%")

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in op_loader:
            pred = model(data).argmax(1)
            correct += pred.eq(target).sum().item()
    n_op = len(op_loader) * op_loader.batch_size
    total_op = 2 * OPERATOR_SAMPLES
    print(f"  Operator accuracy: {100*correct/total_op:.2f}%")

    correct = 0
    with torch.no_grad():
        for data, target in font_loader:
            pred = model(data).argmax(1)
            correct += pred.eq(target).sum().item()
    total_font = 10 * RENDERED_SAMPLES
    print(f"  Rendered font digit accuracy: {100*correct/total_font:.2f}%")
    table = {}
    with torch.no_grad():
        for data, target in font_loader:
            pred = model(data).argmax(1)
            for t, p in zip(target.tolist(), pred.tolist()):
                table.setdefault(t, [0, 0])
                table[t][1] += 1
                table[t][0] += int(t == p)
    print("  Per-class (rendered): " + "  ".join(
        f"{d}:{100*c/n:.0f}%" for d, (c, n) in sorted(table.items())))

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/expr_model.pth")
    print("Saved models/expr_model.pth (12 classes: 0-9, +, -)")


if __name__ == "__main__":
    main()