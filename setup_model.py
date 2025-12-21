import urllib.request
import os
import tarfile
import shutil

# URL des offiziellen Google Coral / TensorFlow Modells
MODEL_URL = "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
LABEL_URL = "https://github.com/google-coral/test_data/raw/master/coco_labels.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def download_file(url, dest_path):
    print(f"⬇️ Lade herunter: {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("✅ Fertig.")
    except Exception as e:
        print(f"❌ Fehler: {e}")

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Ordner erstellt: {MODEL_DIR}")

    # 1. .tflite Datei laden
    model_path = os.path.join(MODEL_DIR, "ssd_mobilenet_v2_coco_quant_postprocess.tflite")
    download_file(MODEL_URL, model_path)

    # 2. Labels laden (damit wir wissen, dass ID 16 = "bird" ist)
    label_path = os.path.join(MODEL_DIR, "coco_labels.txt")
    download_file(LABEL_URL, label_path)

    print("\n🎉 Setup abgeschlossen. Das 'Gehirn' liegt im models/ Ordner.")

if __name__ == "__main__":
    main()

