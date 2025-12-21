import sys
import os
import time
import numpy as np

# Pfad-Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
from src.yolo import YoloDetector
import config

def load_labels():
    """Lädt die COCO Labels für schöne Konsolenausgabe"""
    labels = {}
    label_path = os.path.join(config.BASE_DIR, "models", "coco_labels.txt")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            valid_idx = 0
            for line in lines:
                label = line.strip()
                if label.lower() != "n/a":
                    labels[valid_idx] = label
                    valid_idx += 1
    return labels

def main():
    print(f"🚀 Starte YOLO Performance Benchmark (HEADLESS)...")
    print(f"   Modell: {os.path.basename(config.MODEL_PATH)}")
    print(f"   Auflösung: {config.CAMERA_RES}")
    print(f"   Grayscale: {config.USE_GRAYSCALE}")
    print("-" * 40)

    # 1. Labels laden
    labels = load_labels()

    # 2. Kamera starten
    print("📸 Starte Kamera...")
    cam = CameraStream(resolution=config.CAMERA_RES, fps=config.CAMERA_FPS).start()
    time.sleep(2.0) # Warmup

    # 3. Detektor laden
    print("🧠 Lade Detektor...")
    detector = YoloDetector()
    print("✅ Bereit! Drücke STRG+C zum Beenden.")
    print("-" * 40)

    frame_count = 0
    total_inference_time = 0
    
    try:
        while True:
            # Bild holen
            if config.USE_GRAYSCALE:
                frame = cam.read()
            else:
                frame = cam.read_original()
            
            if frame is None:
                continue

            # --- ZEITMESSUNG START ---
            start_t = time.time()
            
            # Reine Inferenz
            detections = detector.detect_all_objects(frame, config.MIN_CONFIDENCE)
            
            duration = time.time() - start_t
            # --- ZEITMESSUNG ENDE ---

            # Statistik
            inference_ms = duration * 1000
            fps = 1.0 / duration
            
            frame_count += 1
            total_inference_time += duration

            # Ausgabe bauen
            detected_names = []
            for cls_id, score, _ in detections:
                name = labels.get(cls_id, f"ID {cls_id}")
                detected_names.append(f"{name} ({score:.0%})")
            
            obj_str = ", ".join(detected_names) if detected_names else "-"
            
            # Zeile überschreiben (\r), damit das Terminal nicht vollgespammt wird
            # Färbt Vögel rot ein in der Konsole (ANSI Codes), wenn vorhanden
            if any(cls_id in [14, 15, 16] for cls_id, _, _ in detections):
                status_symbol = "🔴 VOGEL ALARM"
            else:
                status_symbol = "🟢 Scan..."

            print(f"\r{status_symbol} | Inferenz: {inference_ms:.1f}ms ({fps:.1f} FPS) | Objekte: {obj_str:<40}", end="")

    except KeyboardInterrupt:
        print("\n\n🛑 Benchmark beendet.")
        if frame_count > 0:
            avg_ms = (total_inference_time / frame_count) * 1000
            avg_fps = frame_count / total_inference_time
            print(f"📊 Durchschnitt: {avg_ms:.1f}ms pro Frame ({avg_fps:.1f} FPS)")
    finally:
        cam.stop()

if __name__ == "__main__":
    main()