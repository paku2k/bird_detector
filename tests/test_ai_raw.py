import sys
import os
import cv2
import time
import numpy as np
from flask import Flask, Response

# Pfad-Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
import config

# Import TFLite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

print("🚀 Starte Raw-KI Test...")

# 1. Kamera starten
cam = CameraStream().start()
time.sleep(2.0)

# 2. KI Manuell laden (ohne ObjectDetector Klasse, damit wir alles sehen)
print("  - Lade TFLite Interpreter...")
interpreter = tflite.Interpreter(model_path=config.MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Labels laden (damit wir wissen was ID 16 ist)
labels = {}
label_path = os.path.join(config.BASE_DIR, "models", "coco_labels.txt")
if os.path.exists(label_path):
    print("  - Lade Labels...")
    with open(label_path, 'r') as f:
        # TFLite Model Labels sind oft zeilenbasiert.
        # Manchmal ist Zeile 0 = ID 0, manchmal ID 1. Wir nehmen an Zeile 0 = ID 0.
        for i, line in enumerate(f.readlines()):
            labels[i] = line.strip()
else:
    print("  ⚠️ Keine Labels gefunden (coco_labels.txt fehlt). Zeige nur IDs.")

def generate_frames():
    while True:
        # Bremse (Full Frame Inferenz ist teurer als kleine Crops!)
        # Wir gehen auf 10 FPS runter, damit der Pi nicht glüht.
        time.sleep(0.1) 

        frame = cam.read()
        if frame is None: continue
        
        h_img, w_img = frame.shape
        
        # --- PREPROCESSING ---
        # 1. Resize auf 300x300 (Modell-Input)
        input_data = cv2.resize(frame, (300, 300))
        
        # 2. Fake-RGB (Grayscale 3x stapeln)
        input_data = cv2.merge([input_data, input_data, input_data])
        input_data = np.expand_dims(input_data, axis=0)

        # --- INFERENZ ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # --- ERGEBNISSE ---
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding Boxen
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Klassen-IDs
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Wahrscheinlichkeiten

        print(f"boxes: {boxes}")
        print(f"classes: {classes}")
        print(f"scores: {scores}")

        # --- VISUALISIERUNG ---
        preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Wir loopen durch alle Ergebnisse
        for i in range(len(scores)):
            score = scores[i]
            
            # Threshold: Zeige alles ab 40% Wahrscheinlichkeit
            if score < 0.4: 
                continue

            # Koordinaten berechnen
            ymin, xmin, ymax, xmax = boxes[i]
            
            left = int(xmin * w_img)
            top = int(ymin * h_img)
            right = int(xmax * w_img)
            bottom = int(ymax * h_img)

            class_id = int(classes[i])
            label_name = labels.get(class_id, f"ID {class_id}")
            
            # Display Text
            text = f"{label_name}: {score:.0%}"
            
            # Farbe je nach Objekt
            color = (0, 255, 0) # Grün Standard
            if "bird" in label_name.lower():
                color = (0, 0, 255) # Rot für Vögel
                text = f"ZILE: {text}"
            elif "person" in label_name.lower():
                color = (255, 0, 0) # Blau für Menschen

            # Zeichnen
            cv2.rectangle(preview, (left, top), (right, bottom), color, 2)
            cv2.putText(preview, text, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Output für Webstream
        ret, buffer = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        print("🎥 Raw-KI Stream läuft auf Port 5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if cam: cam.stop()