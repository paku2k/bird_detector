# TAS: Pigeon Defense System

**TAS** is an automated bird deterrent system based on a Raspberry Pi 4. It utilizes computer vision and machine learning to detect, track, and deter birds using a water gun.

The system is specifically optimized for hardware with limited resources (RPi 4, 1GB RAM) and solves the problem of detecting stationary (perched) birds through a hybrid search strategy.

## Video Demonstration
[videos/bird_detector_1.mp4](https://github.com/paku2k/bird_detector/blob/main/videos/bird_detector_1.mp4)

## Project Objective
Autonomous detection and expulsion of birds within a defined area. The system distinguishes between "moving targets" (incoming flight) and "static targets" (already landed).

## Hardware
* **Controller:** Raspberry Pi 4 Model B (1GB RAM)
* **Sensor:** Pi Camera Module V2 (NoIR)
* **Actuators:** 2x Servos (Pan, Tilt)
* **Trigger:** PIR Motion Sensor (optional, as a wake-up trigger)
* **Pan-Tilt Station:**
<img width="1448" height="1195" alt="Hardware Setup" src="https://github.com/user-attachments/assets/fb6f073a-63f5-48ba-a3a7-94ab76889e4a" />

---

## Software Architecture & Logic

### 1. Vision Engine (Hybrid Search)
Since the NoIR camera has a strong red tint in daylight, the pipeline operates entirely in **Grayscale**. Detection is handled via TensorFlow Lite (MobileNet V2 SSD).

* **Mode A: Motion-Guided (Priority)**
    * Detects motion in the frame (frame differencing).
    * Crops the moving area (Dynamic ROI).
    * Inference runs only on this crop -> *Maximum performance & digital zoom effect.*
* **Mode B: Grid-Scan (Fallback)**
    * If no motion is detected (bird is sitting still), the image is divided into 4 quadrants.
    * Sequential inference on the quadrants -> *Locates sleeping or perched birds.*

### 2. High-Performance Tracking
To reduce CPU load, the system **does not** continue searching via the Neural Network after the initial detection.

* Handover to **OpenCV Tracker** (CSRT or MOSSE).
* Tracks texture/patterns instead of objects -> *Continues to work even if the bird remains still.*
* **Re-Validation:** Every 30 frames, the Neural Network briefly checks if the tracked object is still a bird.

### 3. Ballistics & Servo Control
* PID controller for smooth camera movements.
* **Gravity Compensation:** Target correction (aiming higher) based on distance (estimated via bounding box size).
