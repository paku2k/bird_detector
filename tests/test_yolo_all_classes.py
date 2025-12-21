import sys
import os
import cv2
import time
from flask import Flask, Response

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
from src.yolo import YoloDetector
import config

app = Flask(__name__)

print(f"🚀 Starte YOLO All-Classes Test...")
print(f"   Grayscale Modus: {config.USE_GRAYSCALE}")

# Kamera starten
cam = CameraStream(resolution=config.CAMERA_RES, fps=config.CAMERA_FPS).start()
time.sleep(2.0)

# Detektor laden
detector = YoloDetector()

# COCO Labels laden (für schöne Namen)
LABELS = {}
label_path = os.path.join(config.BASE_DIR, "models", "coco_labels.txt")
if os.path.exists(label_path):
    with open(label_path, "r") as f:
        # WICHTIG: Die coco_labels.txt enthält "n/a" Platzhalter (91 Klassen).
        # YOLO gibt aber IDs von 0-79 zurück (80 Klassen).
        # Wir müssen die "n/a" Zeilen herausfiltern, damit die Indizes übereinstimmen.
        lines = f.readlines()
        valid_idx = 0
        for line in lines:
            label = line.strip()
            # Nur echte Labels speichern, n/a überspringen
            if label.lower() != "n/a":
                LABELS[valid_idx] = label
                valid_idx += 1
else:
    print("⚠️  Keine coco_labels.txt gefunden. Zeige nur IDs.")

def generate_frames():
    while True:
        time.sleep(0.01)

        # Je nach Config holen wir das Bild in Grau oder Farbe
        if config.USE_GRAYSCALE:
            frame = cam.read() # Gibt Grayscale zurück (src/vision.py logic)
        else:
            frame = cam.read_original() # Gibt BGR zurück
        
        if frame is None: continue

        # Kopie für Preview (muss BGR sein zum Zeichnen)
        if len(frame.shape) == 2:
            preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            preview = frame.copy()

        # --- DETECTION ---
        start_t = time.time()
        # Ruft die neue Methode auf, die ALLES zurückgibt
        detections = detector.detect_all_objects(frame, config.MIN_CONFIDENCE)
        fps = 1.0 / (time.time() - start_t)

        # --- ZEICHNEN ---
        for class_id, score, box in detections:
            x, y, w, h = box
            
            # Farbe je nach Klasse: Vogel=Rot, Mensch=Blau, Rest=Grün
            color = (0, 255, 0)
            if class_id in [14, 15, 16]: # Vögel (Standard YOLO ID ist meist 14)
                color = (0, 0, 255)
            elif class_id == 0: # Person (meistens ID 0)
                color = (255, 0, 0)

            # Box
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label_name = LABELS.get(class_id, f"ID {class_id}")
            text = f"{label_name} {score:.0%}"
            
            cv2.putText(preview, text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Info-Overlay
        cv2.putText(preview, f"FPS: {fps:.1f} | Objects: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if config.USE_GRAYSCALE:
             cv2.putText(preview, "GRAYSCALE MODE", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # Stream Output
        preview_small = cv2.resize(preview, (800, 600))
        ret, buffer = cv2.imencode('.jpg', preview_small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cam.stop()