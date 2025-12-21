import cv2
import numpy as np
from config import MOTION_THRESHOLD, MIN_MOTION_AREA

class MotionDetector:
    def __init__(self, accum_weight=0.5):
        # accum_weight: Wie schnell passt sich der Hintergrund an? 
        # 0.5 = sehr schnell (gut für Start), 0.1 = langsam
        self.accum_weight = accum_weight
        self.avg_frame = None

    def detect(self, gray_frame):
        """
        Nimmt ein Grayscale-Frame und gibt die Bounding Box (x, y, w, h)
        der größten Bewegung zurück. Gibt None zurück, wenn keine Bewegung.
        """
        # 1. Weichzeichnen gegen Bildrauschen (Pixel-Flimmern)
        # (21, 21) ist die Größe des Weichzeichners (muss ungerade sein)
        blurred = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # 2. Initialisierung des Hintergrunds (beim allerersten Frame)
        if self.avg_frame is None:
            self.avg_frame = blurred.astype("float")
            return None

        # 3. Hintergrund aktualisieren (Running Average)
        # Wir mischen das aktuelle Bild leicht in den Hintergrund.
        cv2.accumulateWeighted(blurred, self.avg_frame, self.accum_weight)
        
        # 4. Differenz berechnen (Aktuelles Bild MINUS Hintergrund)
        frame_delta = cv2.absdiff(blurred, cv2.convertScaleAbs(self.avg_frame))
        
        # 5. Thresholding: Alles was sich genug geändert hat (über THRESHOLD), wird Weiß.
        thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        
        # 6. Löcher füllen (Dilate): Macht weiße Flecken fetter, damit Lücken verschwinden.
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # 7. Konturen finden (Umrisse der weißen Flecken)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_box = None
        max_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            # Ist die Bewegung groß genug? (kein Rauschen/Wind)
            if area < MIN_MOTION_AREA:
                continue
            
            # Wir suchen nur die ALLERGRÖSSTE Bewegung im Bild (unser Hauptziel)
            if area > max_area:
                max_area = area
                (x, y, w, h) = cv2.boundingRect(c)
                largest_box = (x, y, w, h)
                
        return largest_box