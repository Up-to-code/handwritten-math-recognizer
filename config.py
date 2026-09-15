import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
NUMBERS_DIR = os.path.join(DATA_DIR, "numbers")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CONFIDENCE_THRESHOLD = 0.6
FACE_RECOGNITION_TOLERANCE = 0.5
NUMBER_MODEL_PATH = os.path.join(MODELS_DIR, "number_model.pth")

for d in [DATA_DIR, FACES_DIR, NUMBERS_DIR, PROFILES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
