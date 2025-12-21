import urllib.request
import os

# Wir nutzen ein vortrainiertes YOLOv5-Nano (Int8) Modell
# Quelle: Ein Community-Mirror für RPi Modelle, da Ultralytics nur .pt files direkt hostet
# Dieses Modell erwartet 320x320 Input.
MODEL_URL = "https://github.com/geekcom/yolov5-tflite/raw/main/models/yolov5n-int8-320.tflite"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FILE_NAME = "yolov5n-320.tflite"

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    dest_path = os.path.join(MODEL_DIR, FILE_NAME)
    print(f"⬇️ Lade YOLOv5 Nano (320px, Int8) herunter...")
    
    try:
        urllib.request.urlretrieve(MODEL_URL, dest_path)
        print(f"✅ Download erfolgreich: {dest_path}")
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    download_model()