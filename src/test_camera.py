import cv2
import time
from picamera2 import Picamera2

# ----------------- Kamera Setup -----------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Kamera stabilisieren

print("Kamera Vorschau gestartet. Drücke 'q' zum Beenden.")

# ----------------- Vorschau Schleife -----------------
while True:
    frame = picam2.capture_array()
    if frame is None:
        continue

    # Optional: Graustufen (für IR-Kamera)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Kamera Vorschau", frame)

    # Mit 'q' beenden
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------- Aufräumen -----------------
picam2.stop()
cv2.destroyAllWindows()
print("Vorschau beendet.")
