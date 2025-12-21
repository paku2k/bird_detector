# config.py
import os

# --- PFADE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n_480_int8.tflite")

# --- KAMERA EINSTELLUNGEN ---
CAMERA_RES = (1296, 972) # Niedrige Auflösung für Performance
CAMERA_FPS = 30
FLIP_IMAGE = False      # Falls die Kamera über Kopf hängt

# --- VISION EINSTELLUNGEN ---
MIN_CONFIDENCE = 0.65    # 50% Sicherheit, dass es ein Vogel ist
MOTION_THRESHOLD = 25   # Wie stark muss sich ein Pixel ändern (0-255)
MIN_MOTION_AREA = 500   # Minimale Größe der Bewegung in Pixeln
USE_GRAYSCALE = 1

# --- HARDWARE PINS (Beispiele - später anpassen) ---
# Wir nutzen Board-Nummerierung (GPIO.BCM)
PIN_SERVO_PITCH = 17
PIN_SERVO_ROLL = 27
PIN_TRIGGER = 22
PIN_PIR_SENSOR = 4