# tests/test_setup.py
import sys
import os

# Füge den Hauptordner zum Pfad hinzu, damit wir config importieren können
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import cv2
    import numpy as np
    import RPi.GPIO as GPIO
    import config
    
    print("✅ OpenCV Version:", cv2.__version__)
    print("✅ NumPy Version:", np.__version__)
    print("✅ Config geladen. Kamera Auflösung:", config.CAMERA_RES)
    print("✅ Alles bereit für Phase 2!")
    
except ImportError as e:
    print("❌ Fehler beim Import:", e)
except Exception as e:
    print("❌ Unbekannter Fehler:", e)