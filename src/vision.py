import cv2
import threading
import time
import numpy as np
from picamera2 import Picamera2
from config import CAMERA_RES, CAMERA_FPS, FLIP_IMAGE

class CameraStream:
    def __init__(self, resolution=CAMERA_RES, fps=CAMERA_FPS):
        """
        Initialisiert den Stream.
        :param resolution: Tupel (breite, höhe), überschreibt config.
        :param fps: Gewünschte Framerate, überschreibt config.
        """
        self.picam2 = Picamera2()
        self.resolution = resolution
        
        # 1. Konfiguration für Auflösung
        # Wir nutzen "main" für die Vorschau/Capture mit der gewünschten Auflösung
        config = self.picam2.create_preview_configuration(
            main={"size": self.resolution, "format": "BGR888"}
        )
        self.picam2.configure(config)
        
        # 2. Framerate setzen (WICHTIG!)
        # Picamera2 steuert FPS über "FrameDurationLimits" (in Mikrosekunden)
        # z.B. 30 FPS = 1.000.000 / 30 = 33333 µs pro Frame
        frame_duration_us = int(1_000_000 / fps)
        
        self.picam2.start() # Muss gestartet sein, um Controls zu setzen
        
        # Wir setzen Min und Max gleich, um die FPS zu erzwingen
        # Bei sehr hohen Auflösungen kann die Hardware evtl. nicht mithalten,
        # dann regelt der Pi automatisch runter.
        self.picam2.set_controls({
            "FrameDurationLimits": (frame_duration_us, frame_duration_us),
            # Optional: Noise Reduction und Schärfe für KI verbessern
            "Sharpness": 1.0, 
            "NoiseReductionMode": 2 # 2 = High Quality
        })
        
        # Warm-up und Speicher reservieren
        time.sleep(2.0) # Bei High-Res etwas mehr Zeit geben
        self.frame = self.picam2.capture_array()
        
        if self.frame is None:
            raise IOError("Kamera liefert keine Bilder!")

        # Wir speichern das Original BGR für den Grid-Scan
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.picam2.stop()
                return

            # WARTET HIER hardware-getaktet auf das nächste Bild
            frame = self.picam2.capture_array()
            
            if frame is not None:
                if FLIP_IMAGE:
                    frame = cv2.flip(frame, -1)
                
                # CPU-Intensive Umwandlung sofort hier erledigen
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                self.gray_frame = gray
                self.frame = frame

    def read(self):
        # Gibt das Grayscale-Frame zurück (für Motion Detection & KI)
        return self.gray_frame
    
    def read_original(self):
        # Gibt das Farbbild zurück (für Debugging & Webstream & HD Scan)
        return self.frame

    def stop(self):
        self.stopped = True