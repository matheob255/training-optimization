# 🏃‍♂️ Training Simulator Pro

**An intelligent training plan simulator for endurance athletes** powered by physiological modeling and machine learning.

Analyze your training load, predict injury risk, and optimize your race preparation using data from Garmin Connect.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Features

### 📊 Training Plan Simulation
- **12-week training plan builder** with customizable weekly mileage
- **Real-time injury risk detection** based on ACWR, HRV, and recovery metrics
- **Race performance prediction** for 10K, half-marathon, and marathon
- **Fatigue accumulation modeling** with daily load tracking

### 🔬 Advanced Analytics
- **Acute:Chronic Workload Ratio (ACWR)** monitoring
- **Heart Rate Variability (HRV)** trend analysis
- **Recovery score** calculation based on sleep, stress, and training load
- **Peak fitness detection** - find your optimal race week

### 📈 Data Visualization
- Interactive charts of training load progression
- Week-by-week injury risk heatmaps
- Recovery and readiness trends
- Comparative analysis of different training plans

### 🔐 Secure Data Import
- **Direct Garmin Connect integration** (credentials never stored)
- **CSV file upload** (most secure - no credentials needed)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher (check installation: `python --version in terminal`)
- Garmin Connect account

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/matheob255/training-optimization.git
cd training-optimization
```


2. **Create virtual environment** (recommended)

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venvenv\Scripts\activate
```


3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

5. **Open in browser**
- App opens automatically at `http://localhost:8501`
- If not, navigate to the URL shown in terminal
