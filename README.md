# 🚦 RD-ITAS — Roads Directorate Intersection Traffic Analysis System

> **Automated computer vision pipeline for traffic engineering studies using YOLO, ByteTrack and OpenCV.**

RD-ITAS (Roads Directorate – Intersection Traffic Analysis System) is a Python-based traffic analysis framework developed to automate intersection studies from overhead drone or elevated camera footage.

The system detects, tracks and classifies vehicles and pedestrians, automatically producing the traffic engineering metrics normally collected manually during intersection surveys.

Originally developed for the **Roads Directorate, Kingdom of Lesotho**, the framework is designed to support signal optimization, traffic volume studies, pedestrian analysis and geometric assessment.

---

# Features

## Vehicle Analysis

* Multi-class vehicle detection using YOLO
* Persistent tracking with ByteTrack
* Lane assignment
* Automatic approach counting
* Turning movement classification
* Left / Straight / Right movement matrices
* Passenger Car Unit (PCU) calculations
* Vehicle per hour (VPH) calculations
* Lane-by-lane statistics
* Saturation flow estimation
* Webster equation inputs

---

## Pedestrian Analysis

* Pedestrian detection
* Crossing intent detection
* Waiting zone monitoring
* Crossing time measurement
* Walking speed estimation
* Pedestrian flow rate
* Minimum pedestrian green time estimation

---

## Signal Timing Support

The system integrates traffic counts with signal phase timing to produce engineering parameters including:

* Critical lane volumes
* Saturation flow
* Flow ratios
* Webster optimization inputs
* Phase tagging for every event

---

## Data Validation

A built-in redundancy mechanism validates collected data.

```
Approach Count

        ≈

Left + Straight + Right
```

This allows missing detections caused by occlusion to be identified automatically.

---

# Pipeline

```
Video Input
      │
      ▼
YOLO Detection
      │
      ▼
ByteTrack Tracking
      │
      ▼
Vehicle Classification
      │
      ▼
Approach Line Counting
      │
      ▼
Lane Assignment
      │
      ▼
Intersection Tracking
      │
      ▼
Exit Classification
      │
      ▼
Traffic Engineering Metrics
```

Pedestrians are processed simultaneously through an independent pipeline for crossing analysis.

---

# Outputs

The software automatically generates:

* Live annotated display
* Annotated video (optional)
* CSV event log
* Vehicle approach counts
* Turning movement counts
* PCU/hour
* Vehicle/hour
* Pedestrian statistics
* Saturation flow estimates
* Webster equation inputs

---

# Technologies

* Python
* OpenCV
* Ultralytics YOLO
* ByteTrack
* NumPy

---

# Project Structure

```
RD-ITAS/
│
├── RD_ITAS_main.py
├── RD_ITAS_calibrate.py
├── yolo11n.pt
├── videos/
├── outputs/
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/RD-ITAS.git

cd RD-ITAS
```

Install dependencies

```bash
pip install ultralytics opencv-python numpy
```

---

# Usage

Configure the following parameters:

* Video path
* Calibration distance
* Intersection information
* Signal timing
* Analysis zones

Run the calibration utility

```bash
python RD_ITAS_calibrate.py
```

Then execute the analysis

```bash
python RD_ITAS_main.py
```

Controls:

* **P** → Pause
* **Q** → Quit

---

# Engineering Metrics Produced

* Vehicle counts
* Turning movement counts
* Lane utilization
* Passenger Car Units (PCU)
* Vehicle/hour (VPH)
* Pedestrian/hour
* Average pedestrian speed
* Crossing times
* Saturation flow
* Critical lane flow
* Webster flow ratio
* Signal phase statistics

---

# Future Development

Planned improvements include:

* Multi-camera intersection fusion
* Perspective correction using homography
* Camera auto-calibration
* Automatic lane detection
* Speed estimation using projective geometry
* Queue length estimation
* Delay analysis
* HCM Level of Service calculations
* Automatic traffic signal optimization
* Real-time edge deployment on NVIDIA Jetson and Raspberry Pi AI accelerators

---

# Applications

* Traffic impact assessments
* Signal timing optimization
* Intersection design studies
* Pedestrian safety analysis
* Urban planning
* Smart city deployments
* Transport engineering research

---

# Author

**Kopano Maketekete**

Process Control Systems Engineer

Embedded Systems • Computer Vision • FPGA • AI • Intelligent Transport Systems

---

# License

This project is released under the MIT License.

---

> *Engineering transportation intelligence through computer vision.*
