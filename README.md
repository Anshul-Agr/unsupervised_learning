# V–I Feature Space Pulse Classification

This repository contains a Python script for **automatic classification of electrical discharge pulses** using a **Voltage–Current (V–I) feature space** and unsupervised clustering.

The method is designed for oscilloscope data (time, voltage, current) and classifies pulses into:

* **Normal**
* **Arc**
* **Short**
* **Open / Weak discharge**

based on their physical behavior in the V–I plane.

---

## Core Idea

Each detected pulse is summarized using two physically meaningful features:

* **Average Voltage (V̄)** during the pulse
* **Peak Current (Iₘₐₓ)** during the pulse

These features define a **2D V–I feature space**, where different discharge modes naturally separate.

Unsupervised **K-Means clustering** is then applied, and clusters are **mapped to physical discharge modes** using domain rules.

---

## Input Data Format

The script expects a CSV file with oscilloscope-exported data:

| Column  | Description     |
| ------- | --------------- |
| Time    | Time (seconds)  |
| Current | Current (Amps)  |
| Voltage | Voltage (Volts) |

Notes:

* The script assumes **one header row offset** (`header=1`)
* Data is converted internally to microseconds

Example filename:

```
Scope_Data.csv
```

---

## Processing Pipeline

### 1. Preprocessing

* Convert time to microseconds
* Cast all values to floating point
* Reset indexing

---

### 2. Pulse Detection

Pulses are detected using a **logical trigger**:

```
Active if:
Voltage > V_TRIGGER  OR  Current > I_TRIGGER
```

Default thresholds:

```python
V_TRIGGER = 1.0 V
I_TRIGGER = 50.0 A
```

Short inactive gaps (< 10 μs) are merged to avoid pulse fragmentation.

---

### 3. Feature Extraction (Per Pulse)

For each valid pulse:

| Feature          | Meaning                   |
| ---------------- | ------------------------- |
| Avg_Voltage      | Mean voltage during pulse |
| Max_Current      | Peak current during pulse |
| Duration         | Pulse width (μs)          |
| Start / End Time | For visualization         |

Very short pulses (< 5 μs) are discarded.

---

### 4. V–I Feature Space Clustering

* Features used: **(Avg_Voltage, Max_Current)**
* Min–Max normalization is applied
* **K-Means (k = 4)** is used to discover clusters

Normalization is critical because:

```
Current magnitude >> Voltage magnitude
```

---

### 5. Physics-Based Cluster Labeling

Cluster centers are mapped to discharge modes using **physical rules**:

| Condition                     | Label       |
| ----------------------------- | ----------- |
| Low current                   | Open / Weak |
| High current + Low voltage    | Short       |
| High current + Medium voltage | Arc         |
| High current + High voltage   | Normal      |

This ensures the classification is **interpretable**, not purely statistical.

---

## Outputs

### 1. V–I Scatter Plot

**`vi_scatter.png`**

* X-axis: Peak Current (A)
* Y-axis: Average Voltage (V)
* Each pulse plotted as a point
* Color-coded by discharge type

This plot is the **primary proof** of separation.

---

### 2. Time-Series Classification Plot

**`vi_classification.png`**

* Voltage vs time (top)
* Current vs time (bottom)
* Pulses highlighted using colored spans
* Each pulse annotated with its class label

This confirms **temporal correctness** of classification.

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Execution

```bash
python classify_vi_method.py
```

Make sure the CSV file is in the same directory or update the file path.

---

## File Structure

```
.
├── classify_vi_method.py
├── Scope_Data.csv
├── vi_scatter.png
├── vi_classification.png
└── README.md
```


## Author Notes

This script is intended for **research** of pulsed electrical systems such as:

* EDM
* Spark discharge experiments
* Power electronics fault analysis
