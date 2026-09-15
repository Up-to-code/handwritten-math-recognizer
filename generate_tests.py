import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

FONT_PATH = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"
FALLBACK_FONT_PATH = "/System/Library/Fonts/MarkerFelt.ttc"


def get_font(size):
    path = FONT_PATH if os.path.exists(FONT_PATH) else FALLBACK_FONT_PATH
    return ImageFont.truetype(path, size)


def render_digit(digit, size=280, font_size=190, offset=(0, 0), rotation=0):
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    font = get_font(font_size)
    digit_str = str(digit)
    bbox = draw.textbbox((0, 0), digit_str, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    cx = (size - tw) // 2 + offset[0]
    cy = (size - th) // 2 - bbox[1] + offset[1]
    draw.text((cx, cy), digit_str, fill=0, font=font)

    if rotation:
        img = img.rotate(rotation, fillcolor=255)

    return img


def generate_digit_image(digit, style="clean"):
    size = 280

    if style == "clean":
        img = render_digit(digit, size=size, font_size=np.random.randint(170, 200),
                           offset=(np.random.randint(-20, 20), np.random.randint(-15, 15)))

    elif style == "noisy":
        img = render_digit(digit, size=size, font_size=np.random.randint(170, 200),
                           offset=(np.random.randint(-20, 20), np.random.randint(-15, 15)))
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, 12, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(arr, (3, 3), 1)
        img = Image.fromarray(img)

    elif style == "thick":
        img = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(img)
        font = get_font(np.random.randint(170, 200))
        digit_str = str(digit)
        bbox = draw.textbbox((0, 0), digit_str, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = (size - tw) // 2 - bbox[0] + np.random.randint(-15, 15)
        cy = (size - th) // 2 - bbox[1] + np.random.randint(-15, 15)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                draw.text((cx + dx, cy + dy), digit_str, fill=0, font=font)

    elif style == "thin":
        img = render_digit(digit, size=size, font_size=np.random.randint(130, 160),
                           offset=(np.random.randint(-25, 25), np.random.randint(-20, 20)))
        img = Image.eval(img.rotate(0), lambda x: 255 if x > 245 else int(x * 1.2))

    elif style == "rotated":
        img = render_digit(digit, size=size, font_size=np.random.randint(170, 200),
                           offset=(np.random.randint(-20, 20), np.random.randint(-15, 15)),
                           rotation=np.random.randint(-25, 25))

    elif style == "blurry":
        img = render_digit(digit, size=size, font_size=np.random.randint(170, 200))
        img = img.filter(ImageFilter.GaussianBlur(radius=3))

    elif style == "shifted":
        img = render_digit(digit, size=size, font_size=np.random.randint(170, 200),
                           offset=(np.random.randint(-60, 60), np.random.randint(-55, 55)))

    else:
        img = render_digit(digit, size=size, font_size=190)

    arr = np.array(img, dtype=np.uint8)
    return arr


def generate_test_images(output_dir="data/test_images"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(28)

    test_cases = [
        (3, "clean"),
        (7, "noisy"),
        (5, "thick"),
        (1, "shifted"),
    ]

    generated = []
    for digit, style in test_cases:
        img = generate_digit_image(digit, style)
        filename = f"test_{digit}_{style}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        generated.append({
            "digit": digit,
            "style": style,
            "path": filepath,
            "filename": filename,
        })
        print(f"Generated: {filename} (digit={digit}, style={style})")

    preview = np.zeros((280 * 2 + 40, 280 * 2 + 40, 3), dtype=np.uint8)
    preview[:] = (40, 40, 40)

    for i, g in enumerate(generated):
        img = cv2.imread(g["path"], cv2.IMREAD_GRAYSCALE)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        row = i // 2
        col = i % 2
        y = row * (280 + 20) + 10
        x = col * (280 + 20) + 10

        preview[y:y + 280, x:x + 280] = color_img

        label = f"{g['digit']} ({g['style']})"
        cv2.putText(preview, label, (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    preview_path = os.path.join(output_dir, "preview.png")
    cv2.imwrite(preview_path, preview)
    print(f"\nPreview saved: {preview_path}")

    return generated, preview_path


if __name__ == "__main__":
    print("=" * 50)
    print("  TEST IMAGE GENERATOR")
    print("=" * 50)
    generated, preview = generate_test_images()
    print(f"\nGenerated {len(generated)} test images")
    print(f"Preview: {preview}")