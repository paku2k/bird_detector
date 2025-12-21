import urllib.request
import os

# STABILER LINK: Direkt vom Google Storage (MediaPipe Models)
# Dies ist die Int8 Quantized Version (schnell auf Raspberry Pi)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FILE_NAME = "efficientdet_lite0.tflite"

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    dest_path = os.path.join(MODEL_DIR, FILE_NAME)
    print(f"⬇️ Lade EfficientDet-Lite0 herunter...")
    print(f"   Quelle: {MODEL_URL}")
    
    try:
        # Dieser Google Storage Link braucht meist keine Fake-Headers
        urllib.request.urlretrieve(MODEL_URL, dest_path)
            
        print(f"✅ Download erfolgreich: {dest_path}")
        print("   Größe: {:.2f} MB".format(os.path.getsize(dest_path) / 1024 / 1024))
        print("   Du kannst jetzt config.py anpassen!")
        
    except Exception as e:
        print(f"❌ Fehler beim Download: {e}")
        print(f"   BITTE MANUELL LADEN: {MODEL_URL}")
        print(f"   Speichern unter: {dest_path}")

if __name__ == "__main__":
    download_model()