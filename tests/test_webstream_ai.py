import sys
import os
import cv2
import time
import numpy as np
from flask import Flask, Response

# Pfad-Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
from src.motion import MotionDetector
from src.detector import ObjectDetector
import config

app = Flask(__name__)

# Globale Objekte
cam = None
motion_detector = None
bird_detector = None

def init_system():
    global cam, motion_detector, bird_detector
    print("🚀 Initialisiere System...")
    
    # 1. Kamera
    print("  - Starte Kamera...")
    cam = CameraStream().start()
    time.sleep(2.0) # Warmup
    
    # 2. Motion Detector
    print("  - Lade Bewegungsmelder...")
    motion_detector = MotionDetector()
    
    # 3. AI Detector
    print("  - Lade KI-Modell (Das dauert kurz)...")
    try:
        bird_detector = ObjectDetector()
        print("  ✅ KI geladen!")
    except Exception as e:
        print(f"  ❌ FEHLER beim Laden der KI: {e}")
        sys.exit(1)

def generate_frames():
    while True:
        # Performance-Bremse (20 FPS sind genug)
        time.sleep(0.05) 

        frame = cam.read()
        if frame is None: continue
        
        # Für die Anzeige brauchen wir Farbe
        preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h_img, w_img = frame.shape

        # 1. Bewegung suchen
        motion_box = motion_detector.detect(frame)
        
        if motion_box:
            (x, y, w, h) = motion_box
            
            # 2. Smart Crop (Zoom mit Padding)
            padding = 50
            start_x = max(0, x - padding)
            start_y = max(0, y - padding)
            end_x = min(w_img, x + w + padding)
            end_y = min(h_img, y + h + padding)
            
            # Ausschneiden des ROI (Region of Interest)
            roi = frame[start_y:end_y, start_x:end_x]
            
            # --- DEBUG VISUALISIERUNG START ---
            # Wir zeigen den ROI oben links im Bild an (Bild-in-Bild)
            if roi.size > 0:
                # ROI visuell hervorheben im Hauptbild (grüner Kasten um Scan-Area)
                cv2.rectangle(preview, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
                
                # ROI vergrößern für die Anzeige oben links (z.B. 150x150 Pixel)
                roi_display = cv2.resize(roi, (150, 150))
                roi_display_color = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
                
                # ROI in das Hauptbild kopieren (oben links)
                preview[10:160, 10:160] = roi_display_color
                cv2.rectangle(preview, (10, 10), (160, 160), (255, 255, 0), 2)
                cv2.putText(preview, "KI Input", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # --- DEBUG VISUALISIERUNG ENDE ---

            # 3. KI fragen
            # Debug-Print im Terminal
            print(f"🔎 Scanne ROI {roi.shape}...", end=" ")
            
            is_bird, score, bird_box = bird_detector.detect(roi)
            
            if is_bird:
                print(f"✅ VOGEL! ({score:.2f})")
                
                # Koordinaten umrechnen
                rel_x, rel_y, rel_w, rel_h = bird_box
                final_x = start_x + rel_x
                final_y = start_y + rel_y
                
                # Roter Rahmen für Vogel
                cv2.rectangle(preview, (final_x, final_y), 
                            (final_x + rel_w, final_y + rel_h), (0, 0, 255), 2)
                
                label = f"VOGEL! ({score*100:.0f}%)"
                cv2.putText(preview, label, (final_x, final_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Auch im Debug-Fenster (oben links) den Vogel einzeichnen
                # Umrechnung auf 150x150 Display
                scale_x = 150 / roi.shape[1]
                scale_y = 150 / roi.shape[0]
                d_x = int(rel_x * scale_x) + 10
                d_y = int(rel_y * scale_y) + 10
                d_w = int(rel_w * scale_x)
                d_h = int(rel_h * scale_y)
                cv2.rectangle(preview, (d_x, d_y), (d_x+d_w, d_y+d_h), (0,0,255), 1)
                
            else:
                print("❌ Nichts")
                # Grüner Rahmen um die Motion-Area
                cv2.putText(preview, "Scan...", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
        init_system()
        print("🎥 Webstream läuft auf Port 5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if cam: cam.stop()