import sys
import os
import cv2
import time
import glob
import numpy as np
import csv  # NEU: CSV Modul importiert
from flask import Flask, Response

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
# Wir importieren BEIDE Module, um flexibel zu sein
import src.yolo 
import src.detector 
import config

app = Flask(__name__)

# --- CONFIG ---
TEST_DURATION_SEC = 15
MODELS_DIR = os.path.join(config.BASE_DIR, "models")

# Globale Variablen
benchmark_results = []
current_model_idx = 0
start_time = 0
frame_count = 0
total_inference_time = 0
models_list = []
is_finished = False
active_detector_instance = None

# Statistik Variablen für das aktuelle Modell
bird_frames_count = 0      # In wie vielen Frames war ein Vogel?
bird_conf_accum = 0.0      # Summe der Confidence aller Vogel-Treffer
bird_detections_total = 0  # Anzahl aller erkannten Vögel

# Labels laden (MIT FIX FÜR VERSCHIEBUNG)
labels = {}
label_path = os.path.join(MODELS_DIR, "coco_labels.txt")
if os.path.exists(label_path):
    with open(label_path, "r") as f:
        valid_idx = 0 # Eigener Zähler für lückenlose YOLO-IDs
        for line in f.readlines():
            label_text = line.strip()
            # Wir überspringen n/a und zählen nur echte Labels hoch
            if label_text.lower() != "n/a": 
                labels[valid_idx] = label_text
                valid_idx += 1
else:
    print("⚠️  Keine coco_labels.txt gefunden.")

# ALLE Modelle finden
all_tflite = sorted(glob.glob(os.path.join(MODELS_DIR, "*.tflite")))
if not all_tflite:
    print(f"❌ Keine Modelle in {MODELS_DIR} gefunden!")
    sys.exit(1)

# Liste filtern (optional) oder einfach alle nehmen
models_list = all_tflite

print(f"🚀 Starte Benchmark mit {len(models_list)} Modellen:")
for m in models_list:
    print(f"  - {os.path.basename(m)}")

cam = CameraStream(resolution=config.CAMERA_RES, fps=config.CAMERA_FPS).start()
time.sleep(2.0)

def load_next_model():
    global active_detector_instance, current_model_idx, start_time, frame_count, total_inference_time
    global bird_frames_count, bird_conf_accum, bird_detections_total
    
    if current_model_idx >= len(models_list):
        return False

    model_path = models_list[current_model_idx]
    model_name = os.path.basename(model_path)
    
    print(f"\n🔄 Lade Modell {current_model_idx+1}/{len(models_list)}: {model_name}...")
    
    try:
        # ENTSCHEIDUNG: Welchen Detektor nehmen wir?
        if "yolo" in model_name.lower():
            print("   👉 Nutze YOLO Detector Engine")
            src.yolo.MODEL_PATH = model_path
            active_detector_instance = src.yolo.YoloDetector()
        else:
            print("   👉 Nutze Standard Detector Engine (EfficientDet/SSD)")
            src.detector.MODEL_PATH = model_path
            active_detector_instance = src.detector.ObjectDetector()
            
        print("   ✅ Modell geladen.")
    except Exception as e:
        print(f"   ❌ CRITICAL ERROR beim Laden: {e}")
        active_detector_instance = None

    # Timer & Stats Reset
    start_time = time.time()
    frame_count = 0
    total_inference_time = 0
    bird_frames_count = 0
    bird_conf_accum = 0.0
    bird_detections_total = 0
    
    return True

# Hilfsfunktion zum Speichern der Ergebnisse als CSV
def save_results_to_file():
    filepath = "benchmark_results.csv"
    try:
        # 'newline=""' verhindert leere Zeilen in Excel unter Windows
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f, delimiter=';') # Strichpunkt für Excel (Deutsch) oft besser, sonst ','
            
            # Header
            writer.writerow(["Model Name", "FPS", "Inference Time (ms)", "Bird Detection Rate (%)", "Avg Confidence"])
            
            # Daten
            for res in benchmark_results:
                writer.writerow([
                    res['name'], 
                    f"{res['fps']:.2f}",
                    f"{res['inf_ms']:.2f}", 
                    f"{res['bird_pct']:.2f}", 
                    f"{res['avg_conf']:.4f}"
                ])
                
        print(f"\n💾 Ergebnisse erfolgreich als CSV gespeichert in: {os.path.abspath(filepath)}")
        print("   (Du kannst diese Datei direkt in Excel öffnen)")
    except Exception as e:
        print(f"❌ Fehler beim Speichern der CSV: {e}")

load_next_model()

def generate_frames():
    global current_model_idx, start_time, frame_count, total_inference_time, is_finished
    global bird_frames_count, bird_conf_accum, bird_detections_total
    
    while True:
        if is_finished:
            # Zusammenfassung anzeigen (Tabelle)
            summary_img = np.zeros((600, 1000, 3), dtype=np.uint8) # Breiter für mehr Spalten
            cv2.putText(summary_img, "BENCHMARK COMPLETE - SAVED TO CSV", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Header
            y = 100
            header = f"{'Model':<20} {'FPS':<6} {'Inf(ms)':<8} {'% Bird':<8} {'AvgConf':<8}"
            cv2.putText(summary_img, header, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 20
            cv2.line(summary_img, (20, y), (980, y), (255, 255, 255), 1)
            y += 30
            
            for res in benchmark_results:
                name = res['name']
                if len(name) > 20: name = name[:18] + ".."
                
                fps_str = f"{res['fps']:.1f}"
                inf_str = f"{res['inf_ms']:.0f}"
                bird_pct_str = f"{res['bird_pct']:.0f}%"
                avg_conf_str = f"{res['avg_conf']:.2f}"
                
                # Farbkodierung
                col = (0, 255, 0) if res['fps'] >= 5.0 else (0, 0, 255)
                
                # Spalten manuell positionieren
                cv2.putText(summary_img, name, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(summary_img, fps_str, (280, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                cv2.putText(summary_img, inf_str, (380, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
                cv2.putText(summary_img, bird_pct_str, (500, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(summary_img, avg_conf_str, (620, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                y += 30

            ret, buffer = cv2.imencode('.jpg', summary_img)
            if ret: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1.0)
            continue

        # --- FRAME HOLEN ---
        if getattr(config, 'USE_GRAYSCALE', False):
            frame = cam.read() 
        else:
            frame = cam.read_original() 
            
        if frame is None: 
            time.sleep(0.01)
            continue
        
        if len(frame.shape) == 2:
            preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            preview = frame.copy()

        current_time = time.time()
        elapsed = current_time - start_time
        
        # Zeit abgelaufen?
        if elapsed > TEST_DURATION_SEC:
            # Stats berechnen
            avg_fps = frame_count / elapsed if frame_count > 0 else 0
            avg_inf = (total_inference_time / frame_count) * 1000 if frame_count > 0 else 0
            
            # Neue Bird Stats
            bird_pct = (bird_frames_count / frame_count * 100.0) if frame_count > 0 else 0
            avg_bird_conf = (bird_conf_accum / bird_detections_total) if bird_detections_total > 0 else 0
            
            m_name = os.path.basename(models_list[current_model_idx])
            
            benchmark_results.append({
                "name": m_name, 
                "fps": avg_fps, 
                "inf_ms": avg_inf,
                "bird_pct": bird_pct,
                "avg_conf": avg_bird_conf
            })
            
            print(f"📊 Abschluss {m_name}: {avg_fps:.1f} FPS | Birds: {bird_pct:.1f}% | Ø Conf: {avg_bird_conf:.2f}")

            current_model_idx += 1
            if not load_next_model():
                is_finished = True
                save_results_to_file() # <--- HIER SPEICHERN ALS CSV
                continue
            time.sleep(1.0) 
            continue

        # Inferenz
        detections = []
        if active_detector_instance:
            t_start = time.time()
            try:
                detections = active_detector_instance.detect_all_objects(frame, config.MIN_CONFIDENCE)
            except Exception as e:
                pass
            
            t_dur = time.time() - t_start
            total_inference_time += t_dur
            frame_count += 1
        
        # Stats & Zeichnen
        found_bird_in_frame = False
        
        for class_id, score, box in detections:
            # FILTER: Nur Vögel (IDs 14-16)
            if class_id not in [14, 15, 16]:
                continue

            found_bird_in_frame = True
            bird_detections_total += 1
            bird_conf_accum += score
            
            x, y, w, h = box
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            label = labels.get(class_id, f"ID {class_id}")
            # Confidence anzeigen
            cv2.putText(preview, f"{label}: {score:.0%}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if found_bird_in_frame:
            bird_frames_count += 1

        # Overlay
        model_name = os.path.basename(models_list[current_model_idx])
        time_left = int(TEST_DURATION_SEC - elapsed)
        
        cv2.rectangle(preview, (0, 0), (800, 80), (0, 0, 0), -1)
        cv2.putText(preview, f"TEST: {model_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(preview, f"Time: {time_left}s | Birds Found: {bird_frames_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        if getattr(config, 'USE_GRAYSCALE', False):
             cv2.putText(preview, "GRAYSCALE", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        preview_small = cv2.resize(preview, (800, 600))
        ret, buffer = cv2.imencode('.jpg', preview_small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cam.stop()