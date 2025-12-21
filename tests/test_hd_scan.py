import sys
import os
import cv2
import time
import numpy as np
from flask import Flask, Response

# Pfad-Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
from src.detector import ObjectDetector
import config

app = Flask(__name__)

# --- CONFIG ---
# Hohe Auflösung für Details (Pi Cam v2 Full FOV Binning Mode)
SCAN_RES = (1640, 1232) 
SCAN_FPS = 5

print(f"🚀 Starte HD-Scan Test ({SCAN_RES[0]}x{SCAN_RES[1]})...")

# 1. Kamera starten
cam = CameraStream(resolution=SCAN_RES, fps=SCAN_FPS).start()
time.sleep(2.0)

# 2. KI laden
detector = ObjectDetector()
print(f"✅ KI bereit ({detector.input_w}x{detector.input_h}).")

def sliding_window(image, step_size, window_size):
    """Generator, der Kacheln (Crops) aus dem Bild liefert"""
    w_win, h_win = window_size
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # Sicherstellen, dass wir nicht über den Rand hinausgehen
            # (Wir schneiden den Rest einfach ab oder nehmen kleinere Kacheln am Rand)
            yield (x, y, image[y:y + h_win, x:x + w_win])

def generate_frames():
    while True:
        frame = cam.read_original() # Farbe ist wichtig!
        if frame is None: continue

        preview = frame.copy()
        
        # Wir holen die Kachel-Größe dynamisch vom geladenen Modell (EfficientDet = 320, MobileNet = 300)
        tile_w = detector.input_w
        tile_h = detector.input_h
        
        # Step Size: Wir lassen die Kacheln überlappen (Overlap), 
        # damit ein Vogel, der genau auf der Kante sitzt, nicht übersehen wird.
        # 20% Overlap ist ein guter Wert.
        step_size = int(tile_w * 0.8) 
        
        found_birds = []
        
        # --- DER SCAN LOOP ---
        for (x, y, window) in sliding_window(frame, step_size, (tile_w, tile_h)):
            
            # Rand-Check: Die KI braucht exakte Größen.
            # Wenn wir am Rand sind und die Kachel kleiner ist, padden wir sie oder überspringen.
            # Einfachste Lösung: Überspringen wenn zu klein (am Rand sitzt meist eh nichts relevantes)
            if window.shape[0] != tile_h or window.shape[1] != tile_w:
                continue

            # Inferenz mit Fehlerschutz
            try:
                is_bird, score, box = detector.detect(window)
            except Exception as e:
                # Falls die KI hier crasht (z.B. wegen Input-Format), ignorieren wir diese Kachel
                # und machen mit der nächsten weiter, statt den Server zu killen.
                # print(f"Scan-Fehler bei Kachel {x},{y}: {e}") 
                continue
            
            # Debug: Raster einzeichnen (dunkelgrau)
            cv2.rectangle(preview, (x, y), (x + tile_w, y + tile_h), (40, 40, 40), 1)

            if is_bird:
                # Koordinaten umrechnen (Relativ zur Kachel -> Relativ zum Gesamtbild)
                rel_x, rel_y, rel_w, rel_h = box
                global_x = x + rel_x
                global_y = y + rel_y
                
                found_birds.append((global_x, global_y, rel_w, rel_h, score))

                # ROTER KASTEN (TREFFER)
                cv2.rectangle(preview, (global_x, global_y), 
                            (global_x + rel_w, global_y + rel_h), (0, 0, 255), 4)
                
                cv2.putText(preview, f"VOGEL {score:.0%}", (global_x, global_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                print(f"✅ VOGEL GEFUNDEN bei {global_x},{global_y} ({score:.2f})")
        
        if len(found_birds) > 0:
            cv2.putText(preview, f"{len(found_birds)} VOGEL ERKANNT!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        
        # Bild verkleinern für den Webstream
        preview_small = cv2.resize(preview, (820, 616))
        
        ret, buffer = cv2.imencode('.jpg', preview_small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cam.stop()