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

# Wir nutzen die Auflösung aus der Config (die wir dort auf 1280x960 hochgesetzt haben)
# FPS begrenzen, da große Modelle langsamer sind
print(f"🚀 Starte YOLO Full-Frame Test...")
print(f"   Kamera: {config.CAMERA_RES}")
print(f"   Modell: {config.MODEL_PATH}")

cam = CameraStream(resolution=config.CAMERA_RES, fps=config.CAMERA_FPS).start()
time.sleep(2.0)

# Lade Detektor
detector = YoloDetector()

def generate_frames():
    while True:
        # Bremse, um CPU zu schonen, wenn die KI schneller ist als gedacht
        # Bei einem 640er Modell auf dem Pi 4 rechnen wir mit ca 2-4 FPS.
        time.sleep(0.01)

        frame = cam.read_original()
        if frame is None: continue

        # Wir arbeiten auf einer Kopie für die Anzeige
        preview = frame.copy()
        
        # --- ZEITMESSUNG START ---
        start_time = time.time()
        
        # DIREKTE ERKENNUNG AUF DEM GANZEN BILD
        # Der Detector skaliert es intern runter auf 640x640 (oder was das Modell braucht),
        # rechnet aber die Koordinaten für uns wieder hoch auf 1280x960.
        is_bird, score, box = detector.detect(frame)
        
        inference_time = (time.time() - start_time) * 1000 # ms
        
        # --- ZEITMESSUNG ENDE ---

        # FPS Anzeige
        cv2.putText(preview, f"Inferenz: {inference_time:.0f}ms", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if is_bird:
            x, y, w, h = box
            
            # ROTER KASTEN
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 4)
            
            label = f"VOGEL {score:.0%}"
            cv2.putText(preview, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            print(f"✅ VOGEL! ({score:.2f}) bei {x},{y}")
        
        # Bild verkleinern für den Webstream (Bandbreite sparen)
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