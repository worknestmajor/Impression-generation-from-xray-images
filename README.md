# Summarized Medical Report Generation on Chest X-Rays

## Overview
Medical imaging is a critical diagnostic tool in modern healthcare. Interpreting chest X-rays and writing clinical reports manually can take 5–10 minutes per examination, placing a significant workload on radiologists who interpret hundreds of scans daily.

This project implements an automated deep learning pipeline that generates the **Impression** section of a radiology report directly from chest X-ray images (frontal and lateral views). Using an encoder-decoder architecture with CheXNet feature extraction and Bahdanau additive attention, the system produces concise, clinically relevant diagnostic impressions along with visual attention heatmaps indicating where the model focused for each generated word.

---

## Features
- **Dual-View Support**: Accepts frontal and optional lateral chest X-ray images.
- **Attention Visualization**: Interactive token-by-token spatial attention heatmap overlays on both input views.
- **Export Capabilities**: Download generated impressions as formatted PDF or JSON reports.
- **Session Management**: Tracks recent reports and examination history across user sessions.
- **Interactive UI**: Clean, multi-tab Streamlit interface (`Upload X-rays`, `Try a sample`, and `Recent reports`).

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/fastio19/Summarized-Medical-Report-Generation-on-Chest-X-Rays.git
cd Summarized-Medical-Report-Generation-on-Chest-X-Rays
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell / Command Prompt):**
```bash
python -m venv env
.\env\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run final.py
```

Once started, open your web browser at `http://localhost:8501` to use the application.

