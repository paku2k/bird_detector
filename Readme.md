---
layout: default
title: Portfolio | TAS Project
---

# [Your Name]
**Robotics & Computer Vision Enthusiast** [GitHub](https://github.com/yourhandle) | [LinkedIn](https://linkedin.com/in/yourhandle) | [Email](mailto:you@example.com)

---

## 🤖 Featured Project: TAS (Pigeon Defense System)

**TAS** is an automated avian deterrence system designed to protect outdoor areas. Built on a Raspberry Pi 4, it integrates real-time object detection, a hybrid tracking pipeline, and a precision water-turret actuator.

### 🎥 System Demonstration
<video width="100%" height="auto" autoplay loop muted playsinline style="border-radius: 8px; border: 1px solid #ddd;">
  <source src="./videos/bird_detector_1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### 🛠️ Hardware Integration
The system is designed for high-torque responsiveness and reliability in outdoor conditions.

* **Compute:** Raspberry Pi 4 Model B (1GB RAM)
* **Optics:** Pi Camera Module V2 (NoIR)
* **Actuation:** Dual-Servo Pan-Tilt Assembly
* **Mechanism:** Integrated pump and solenoid trigger
* **Design:** 

---

### 🧠 Software Architecture

The primary challenge was achieving high-speed inference on a CPU-only system with limited memory. I solved this by implementing a multi-stage vision pipeline:

#### 1. Hybrid Vision Engine
To mitigate the infrared color shift of the NoIR sensor, the pipeline processes frames in **Grayscale** using **TensorFlow Lite (MobileNet V2 SSD)**.
* **Motion-Guided Mode:** The system identifies movement via frame differencing to extract a Dynamic ROI. Inference is localized to this region, significantly increasing FPS.
* **Grid-Scan Mode (Fallback):** When a target is stationary, the system systematically scans quadrants to detect "perched" birds that would otherwise be ignored by motion sensors.

#### 2. High-Performance Tracking
To minimize CPU overhead, the Neural Network is only used for initial acquisition and periodic re-validation.
* **Correlation Tracking:** Once a target is locked, the system switches to an **OpenCV Tracker (MOSSE/CSRT)**.
* **Persistence:** This allows the system to maintain a lock even if the bird stops moving or changes orientation.

#### 3. Control & Ballistics
Targeting is governed by a **PID Controller** to ensure smooth tracking without servo jitter.
* **Gravity Compensation:** The system estimates distance based on the bounding box area and applies a vertical offset to the pitch servo to account for projectile arc.

---

## 🏗️ Technical Skills & Interests

### Engineering & DIY
* **Digital Fabrication:** Proficiency in Fusion 360 for 3D-printed mechanical housings and structural components.
* **Electronics:** Experience with ESP32/Arduino microcontrollers, sensor fusion, and custom PCB soldering.
* **Rapid Prototyping:** I enjoy the "fail fast" approach—building physical mockups to test mechanical limits before final assembly.

### Outdoors & Resilience
* **Alpine Hiking & Navigation:** I frequently undertake multi-day outdoor trips. The logistical planning and "on-the-fly" problem solving required in the wilderness directly translate to my persistence when debugging complex robotic systems.

---
