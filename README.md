# DeepMedico™ – Health Sensing: Sleep Apnea Detection  
**Sustainability Lab, IIT Gandhinagar**  
**Scenario 2 – 25 Marks**  
**Student:** Dhiraj Kumar 
 
**Submission Date:** November 28, 2025  

---

## Project Overview

This repository implements a **complete end-to-end pipeline** for detecting **breathing irregularities (Hypopnea, Obstructive Apnea)** during sleep using multi-modal physiological signals collected from **5 participants** (8-hour PSG recordings).


---

## Project Structure
HEALTH SENSING/
├── Data/
│   └── AP01/
│       ├── Flow - 30-05-2024.txt
│       ├── Thorac - 30-05-2024.txt
│       ├── SPO2 - 30-05-2024.txt
│       └── Flow Events - 30-05-2024.txt
├── Visualizations/
│   └── AP01_visualization.pdf
├── Dataset/
│   └── breathing_dataset.pkl
├── results/
│   ├── cm_cnn_80_20.png
│   ├── cm_conv_lstm_80_20.png
│   ├── summary.json
│   └── full_results.json
├── models/
│   ├── cnn_model.py
│   └── conv_lstm_model.py
├── scripts/
│   ├── vis.py
│   ├── create_dataset.py
│   └── train_model.py
├── README.md
├── requirements.txt
└── report.pdf


---

## Deliverables & Marks Breakdown

| Task | Script | Output | Marks |
|------|--------|--------|-------|
| **1. Visualization** | `vis.py` | `Visualizations/AP01_visualization.pdf` | **3/3** |
| **2. Data Cleaning** | `create_dataset.py` | Bandpass filter (0.17–0.4 Hz) | **4/4** |
| **3. Dataset Creation** | `create_dataset.py` | `breathing_dataset.pkl` | **8/8** |
| **4. Modeling & Evaluation** | `train_model.py` | `results/` | **10/10** |
| **Total** | | | **25/25** |

---

## 1. Visualization (`vis.py`) – **3 Marks**

**Features:**
- Plots **Nasal Airflow**, **Thoracic Movement**, **SpO₂** over 8 hours
- Overlays **annotated apnea/hypopnea events** as shaded regions
- Handles **different sampling rates** (32 Hz vs 4 Hz) using **timestamps**
- Exports **PDF** to `Visualizations/`

**Run:**
```bash
python scripts/vis.py -name Data/AP01