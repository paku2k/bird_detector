import sys
import os
import cv2
import time
import threading
import queue
import numpy as np
from flask import Flask, Response

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
from src.yolo import YoloDetector
from src.tracking import ObjectTracker
import config

app = Flask(__name__)

# --- CONFIG ---
REVALIDATE_INTERVAL = 6.0  # Alle 2 Sekunden prüft die KI den Tracker
MAX_MISSED_FRAMES = 3      # Toleranz wenn KI nichts sieht

# --- ZUSTANDSMASCHINE ---
STATE_SEARCHING = 0
STATE_TRACKING = 1
current_state = STATE_SEARCHING

# Globale Variablen
latest_frame = None 
detection_result_queue = queue.Queue()
is_running = True

def ai_worker():
    """Intelligenter KI-Thread"""
    global latest_frame, is_running, current_state
    
    print("🧠 KI-Thread gestartet.")
    detector = YoloDetector()
    
    last_check_time = 0
    
    while is_running:
        # Wir holen uns eine Kopie des aktuellen Frames
        if latest_frame is None:
            time.sleep(0.1)
            continue
            
        current_time = time.time()
        
        # ENTSCHEIDUNG: Soll die KI rechnen?
        should_run = False
        
        # LOGIK: Wir lassen die KI öfter laufen, um den Tracker zu validieren
        if current_state == STATE_SEARCHING:
            should_run = True
            sleep_time = 0.05
        elif current_state == STATE_TRACKING:
            # Im Trackmodus: Intervall-Check
            if (current_time - last_check_time) > REVALIDATE_INTERVAL:
                should_run = True
                last_check_time = current_time
                sleep_time = 0.1
            else:
                sleep_time = 0.1
        else:
            sleep_time = 0.1
        
        if should_run:
            # Wir arbeiten auf einer Kopie
            img_for_ai = latest_frame.copy()
            try:
                found, score, box = detector.detect(img_for_ai)
                # Ergebnis in Queue legen
                detection_result_queue.put((found, score, box))
            except Exception as e:
                print(f"KI Fehler: {e}")
        
        time.sleep(sleep_time)

def generate_frames():
    global latest_frame, current_state
    
    tracker = ObjectTracker(tracker_type="CSRT")
    
    # Status Variablen
    last_yolo_box = None
    tracker_box = None
    missed_yolo_counter = 0 
    
    last_processed_frame_id = 0

    while True:
        # Grayscale-Option aus Config beachten
        if getattr(config, 'USE_GRAYSCALE', False):
            frame = cam.read() 
        else:
            frame = cam.read_original() 
            
        if frame is None: 
            time.sleep(0.01)
            continue
        
        # CPU Saver: Nur neue Frames verarbeiten
        if id(frame) == last_processed_frame_id:
            time.sleep(0.01) 
            continue
        
        last_processed_frame_id = id(frame)

        latest_frame = frame.copy()
        
        preview = latest_frame.copy()
        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        # ---------------------------------------------------------
        # 1. KI ERGEBNISSE PRÜFEN (ASYNC)
        # ---------------------------------------------------------
        try:
            while not detection_result_queue.empty():
                ai_result = detection_result_queue.get_nowait()
                found_ai, score_ai, box_ai = ai_result
                
                if found_ai:
                    last_yolo_box = box_ai
                    missed_yolo_counter = 0
                    
                    # FALL A: Wir suchen noch -> Sofort übernehmen
                    if current_state == STATE_SEARCHING:
                        print(f"🚀 YOLO Treffer -> Starte Tracker direkt bei {box_ai}")
                        tracker.start(frame, box_ai)
                        current_state = STATE_TRACKING
                        tracker_box = box_ai

                    # FALL B: Wir tracken schon -> Validierung
                    elif current_state == STATE_TRACKING and tracker_box is not None:
                        # Wir vergleichen einfach die aktuelle Tracker-Pos mit der (verzögerten) KI-Pos.
                        # Bei sitzenden Vögeln passt das.
                        iou = ObjectTracker.calculate_iou(tracker_box, box_ai)
                        
                        # Wenn Überlappung zu gering ist, vertrauen wir der KI mehr als dem Tracker
                        if iou < 0.3: 
                            print(f"⚠️ DRIFT (IoU={iou:.2f}) -> Hard Reset auf YOLO Position!")
                            tracker.start(frame, box_ai)
                            tracker_box = box_ai
                        else:
                            # print(f"✅ Tracker bestätigt (IoU={iou:.2f})")
                            pass

                else:
                    # KI hat nichts gefunden
                    if current_state == STATE_TRACKING:
                        missed_yolo_counter += 1
                        # print(f"⚠️ YOLO sieht nichts ({missed_yolo_counter}/{MAX_MISSED_FRAMES})")
                        
                        if missed_yolo_counter >= MAX_MISSED_FRAMES:
                            print("❌ Vogel verloren (KI bestätigt). Stop.")
                            tracker.stop()
                            current_state = STATE_SEARCHING
                            missed_yolo_counter = 0

        except queue.Empty:
            pass

        # ---------------------------------------------------------
        # 2. TRACKER UPDATE (AKTUELL)
        # ---------------------------------------------------------
        if current_state == STATE_TRACKING:
            success, t_box = tracker.update(frame)
            
            if success:
                tracker_box = t_box
                x, y, w, h = t_box
                cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(preview, "LOCKED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Fadenkreuz
                cx, cy = x + w//2, y + h//2
                cv2.circle(preview, (cx, cy), 5, (0, 255, 0), -1)
            else:
                current_state = STATE_SEARCHING

        # Debug & Output
        if current_state == STATE_SEARCHING:
            cv2.putText(preview, "SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if last_yolo_box:
                # Zeige Geisterbox wo KI zuletzt was gesehen hat
                lx, ly, lw, lh = last_yolo_box
                cv2.rectangle(preview, (lx, ly), (lx+lw, ly+lh), (255, 0, 0), 1)

        preview_small = cv2.resize(preview, (800, 600))
        ret, buffer = cv2.imencode('.jpg', preview_small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    cam = CameraStream(resolution=config.CAMERA_RES, fps=config.CAMERA_FPS).start()
    time.sleep(2.0)

    ai_thread = threading.Thread(target=ai_worker)
    ai_thread.daemon = True
    ai_thread.start()

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        is_running = False
        cam.stop()