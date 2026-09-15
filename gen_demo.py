import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import draw_pad

FONT = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"


def glyph(ch, target_h=70):
    font = ImageFont.truetype(FONT, 240)
    img = Image.new("L", (240, 240), 255)
    dr = ImageDraw.Draw(img)
    bx = dr.textbbox((0, 0), ch, font=font)
    tw, th = bx[2] - bx[0], bx[3] - bx[1]
    dr.text(((240 - tw) // 2 - bx[0], (240 - th) // 2 - bx[1]), ch, fill=0, font=font)
    a = np.array(img)
    xs = np.nonzero((255 - a).sum(axis=0))[0]
    ys = np.nonzero((255 - a).sum(axis=1))[0]
    c = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = target_h / c.shape[0]
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), int(c.shape[0] * s)))


def draw_expression(app, text, gap=30, target_h=70):
    app.canvas[:] = 255
    app.number = None
    app.digits = []
    x = 30
    for ch in text:
        c = glyph(ch, target_h=target_h)
        h, w = c.shape
        if x + w > 470:
            break
        gray = cv2.cvtColor(app.canvas, cv2.COLOR_BGR2GRAY)
        gray[70:70 + h, x:x + w] = cv2.min(gray[70:70 + h, x:x + w], c)
        app.canvas[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        x += w + gap
    app.live_predict()


app = draw_pad.DrawPad()

draw_expression(app, "12+5", gap=32, target_h=72)
frame = app.draw_ui()
cv2.imwrite("docs/screenshot_expression.png", frame)
print("expression 12+5 ->", app.number, "=", app.result,
      "tokens", [d["token"] for d in app.digits])

draw_expression(app, "60938", gap=24, target_h=50)
frame = app.draw_ui()
cv2.imwrite("docs/screenshot_bignum.png", frame)
print("bignum 60938 ->", app.number,
      "tokens", [d["token"] for d in app.digits])

print("saved docs/screenshot_expression.png and docs/screenshot_bignum.png")