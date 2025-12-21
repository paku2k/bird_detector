# tests/test_camera.py
import sys
import os
import cv2
import time

# Fügt das Parent-Directory zum PYTHONPATH hinzu, damit wir 'src' importieren können
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream

def test_stream():
    print("🎥 Starte Kamera-Stream (Threaded)...")
    
    # Instanziierung startet sofort den Thread
    cam = CameraStream().start()
    
    # Warm-Up Zeit: Der Kamerasensor muss Auto-Exposure (Belichtung) 
    # und Auto-White-Balance (AWB) einregeln. Das erste Bild ist oft schwarz oder grün.
    time.sleep(1.0)
    
    print("✅ Stream läuft. Drücke 'q' zum Beenden.")
    
    prev_time = time.time()
    
    try:
        while True:
            # Non-blocking Call! Das geht im Mikrosekunden-Bereich.
            frame = cam.read()
            
            # Sicherheitscheck: Am Anfang kann frame noch 'None' sein,
            # bevor der Thread den ersten Loop beendet hat.
            if frame is None:
                continue

            # Zeige das Bild an. 
            # WICHTIG: 'imshow' ist langsam und blockierend. 
            # In der finalen Version (Headless) kommt das raus!
            cv2.imshow("NoIR Grayscale Feed", frame)
            
            # FPS Berechnung "On the fly"
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Debug Print im Terminal (überschreibt die Zeile mit \r)
            print(f"FPS: {fps:.1f} | Shape: {frame.shape}", end='\r')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nBeende durch Nutzer...")
        
    finally:
        # Cleanup: Wichtig, damit /dev/video0 freigegeben wird.
        cam.stop()
        cv2.destroyAllWindows()
        print("\nKamera geschlossen.")

if __name__ == "__main__":
    test_stream()