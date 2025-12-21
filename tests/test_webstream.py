import sys
import os
import cv2
import time
import numpy as np # Wichtig für das Zusammenkleben der Bilder
from flask import Flask, Response

# Pfad-Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
# Wir implementieren die Logik hier manuell zur Visualisierung, 
# daher brauchen wir MotionDetector import nicht zwingend, aber wir nutzen Config
import config

app = Flask(__name__)

print("🎥 Starte Kamera-Thread...")
try:
    cam = CameraStream().start()
    time.sleep(2.0) # Warten auf Warmup
except Exception as e:
    print(f"KRITISCHER FEHLER: {e}")
    sys.exit(1)

# Variablen für die Motion Detection Logik
avg_frame = None
# Wir nutzen die Werte aus der Config
WEIGHT = 0.4 

def generate_frames():
    global avg_frame
    while True:
        # Bremse
        time.sleep(0.05) 

        frame = cam.read()
        if frame is None: continue

        # --- SCHRITT 1: Weichzeichnen (Blur) ---
        # Entfernt Rauschen, damit nicht jeder Pixelfehler als Bewegung zählt
        gray = frame # Input ist bereits Grayscale
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # --- SCHRITT 2: Hintergrund-Modell (Average) ---
        if avg_frame is None:
            avg_frame = blurred.astype("float")
            continue
        
        # Den Hintergrund langsam an das neue Bild anpassen
        cv2.accumulateWeighted(blurred, avg_frame, WEIGHT)
        
        # --- SCHRITT 3: Delta (Differenz) ---
        # Unterschied berechnen: |Aktuell - Hintergrund|
        frame_delta = cv2.absdiff(blurred, cv2.convertScaleAbs(avg_frame))

        # --- SCHRITT 4: Threshold (Maske) ---
        # Alles was sich stark genug geändert hat wird Weiß (255), Rest Schwarz (0)
        thresh = cv2.threshold(frame_delta, config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2) # Lücken füllen

        # --- SCHRITT 5: Konturen finden & Zeichnen ---
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Wir malen auf eine Bunt-Kopie des Originals
        preview_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        for c in cnts:
            if cv2.contourArea(c) < config.MIN_MOTION_AREA:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- VISUALISIERUNG ZUSAMMENBAUEN (2x2 Grid) ---
        
        # Wir konvertieren alle Zwischenschritte in BGR (Farbe), 
        # damit wir Text draufschreiben und sie stapeln können.
        
        # 1. Oben Links: Ergebnis
        tl = preview_frame
        cv2.putText(tl, "1. Ergebnis", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Oben Rechts: Hintergrund (Was der Pi als 'Normal' sieht)
        tr = cv2.cvtColor(cv2.convertScaleAbs(avg_frame), cv2.COLOR_GRAY2BGR)
        cv2.putText(tr, "2. Hintergrund (Lernend)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 3. Unten Links: Delta (Der reine Unterschied)
        bl = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        cv2.putText(bl, "3. Delta (Unterschied)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 4. Unten Rechts: Threshold (Die finale Entscheidung)
        br = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.putText(br, "4. Threshold (Maske)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Bilder verkleinern, damit das Grid nicht riesig wird (auf 320x240 pro Bild)
        h_grid, w_grid = 240, 320
        tl = cv2.resize(tl, (w_grid, h_grid))
        tr = cv2.resize(tr, (w_grid, h_grid))
        bl = cv2.resize(bl, (w_grid, h_grid))
        br = cv2.resize(br, (w_grid, h_grid))

        # Zusammenkleben: Erst Zeilen, dann Spalten
        top_row = np.hstack([tl, tr])
        bot_row = np.hstack([bl, br])
        grid = np.vstack([top_row, bot_row])

        # --- Output für Web ---
        ret, buffer = cv2.imencode('.jpg', grid, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        print("Stoppe Kamera...")
        cam.stop()