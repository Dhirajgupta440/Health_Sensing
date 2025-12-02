# 🩺 Health Sensing: Breathing Irregularity Detection

This repository contains the full implementation of **Scenario-2: Health Sensing** for the IIT Gandhinagar — Sustainability Lab Internship Task 2025.

Goal → Detect abnormal breathing events (Apnea/Hypopnea) from overnight physiological recordings (8 hours × 5 participants).

---

## 📊 Dataset Details

| Signal | Sampling Rate |
|--------|--------------|
| Nasal Airflow | 32 Hz |
| Thoracic Movement | 32 Hz |
| SpO₂ | 4 Hz |

Annotations Provided:
- Flow Events → Hypopnea, Obstructive Apnea
- Sleep Profile → Sleep Stages (Wake, REM, N1, N2, N3)

---

## 🔗 Important Links

| Description | Link |
|------------|------|
| 📌 Raw Dataset | https://drive.google.com/drive/folders/1AU1rhcpZiUilQy7fYpWQXE0s3Mad5nsY |
| 📌 Results & Outputs | https://drive.google.com/drive/folders/14ETYATkbmjHWyHRMzCrS4M0dpQI2FHAr |

Due to size limits, raw data is stored externally.

---


---

## 🧪 Tasks & Outcomes

✔ Visualization of signals with event markers  
✔ Noise reduced using band-pass filtering (0.17–0.4 Hz)  
✔ Dataset created using 30s windows (50% overlap)  
✔ Three Classes: Normal / Hypopnea / OSA  
✔ LOPO-Cross Validation ✓  
✔ Per-class performance results ✓  

---

## 🚀 Usage

### 🔹Visualization
```bash
python scripts/vis.py -name "AP01"
