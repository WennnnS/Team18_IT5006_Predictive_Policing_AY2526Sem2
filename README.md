# Team18_IT5006_Predictive_Policing_AY2526Sem2

Predictive Policing Project – IT5006 (AY2025/26 Semester 2)

## Project Overview

This project studies short-term crime hotspot prediction using historical Chicago crime data.  
The work was completed in three stages:

- **Phase 1**: exploratory data analysis (EDA) and dashboard prototyping
- **Phase 2**: feature engineering, model building, model evaluation, and model comparison
- **Phase 3**: proof-of-concept deployment as an interactive Streamlit application

Rather than predicting individual incidents, the project focuses on a more practical and stable task:

**predict whether a Chicago police district will become a hotspot in the next 4-hour block.**

This is formulated as a **binary classification** problem at the district × date × time-block level.

---

## Project Scope by Phase

### Phase 1: EDA and dashboard
Phase 1 focused on understanding the structure of the Chicago crime dataset and identifying useful spatial-temporal patterns.

Main work included:
- temporal trend analysis
- spatial hotspot exploration
- crime-type correlation analysis
- initial interactive Streamlit dashboard

### Phase 2: modeling and evaluation
Phase 2 converted the raw event-level dataset into a district-level panel and built hotspot prediction models.

Main work included:
- data cleaning and preprocessing
- district × date × 4-hour block panel construction
- hotspot label construction
- lag and rolling feature engineering
- training and evaluation of multiple classifiers
- threshold tuning and model comparison

Models explored:
- Logistic Regression
- Random Forest
- Gradient Boosting
- HistGradientBoosting

### Phase 3: deployment
Phase 3 deployed the model as a **POC/MVP Streamlit application**.

The final app demonstrates:
- single-record hotspot prediction
- daily risk scanning and ranking
- basic usage guidance
- basic validation and error handling

---

## Prediction Task

The deployed prediction task is:

**Given the current information for a Chicago police district and 4-hour time block, predict whether that district will be a hotspot in the next 4-hour block.**

### Prediction unit
- spatial unit: **Chicago police district**
- temporal unit: **4-hour block**
- target: **next-block hotspot / not hotspot**

### Time block mapping
- Block 0 → 00:00–03:59
- Block 1 → 04:00–07:59
- Block 2 → 08:00–11:59
- Block 3 → 12:00–15:59
- Block 4 → 16:00–19:59
- Block 5 → 20:00–23:59

---

## Dataset

### Primary source
Chicago crime records covering the 2015–2025 analytical window.

### Main split
- **Training period**: 2015–2024
- **Test / demonstration period**: 2025

### Featured deployment dataset
The deployment app uses the processed Phase 2 feature table rather than raw event-level input.

Key fields include:
- `district`
- `event_date`
- `time_block`
- `crime_count`
- `hotspot_current`
- lag features
- rolling mean features
- `target_hotspot_next_block`

---

## Model Selection for Deployment

The deployment app uses:

**HistGradientBoosting**

Why this model was selected for deployment:
- it performed strongly in Phase 2 evaluation
- it is much lighter for deployment than the Random Forest artifact
- it supports a compact and stable Streamlit deployment workflow

### Deployment threshold
The deployed operating threshold is:

**0.28**

This threshold is used instead of the default 0.50 because Phase 2 evaluation showed that HistGradientBoosting performed better under a tuned threshold setting.

---

## Deployment Features

The final Streamlit deployment includes three main pages:

### 1. Overview
Provides:
- project summary
- deployment setup
- usage notes
- time-block mapping
- basic validation notes

### 2. Single Prediction
Allows the user to:
- inspect one district/date/time-window record
- run a next-block hotspot prediction
- view probability, label, and risk interpretation
- use fixed demo cases for stable presentation

### 3. Daily Risk Scan
Allows the user to:
- scan multiple districts on a selected date
- rank district/time-window combinations by predicted hotspot risk
- view top risk records
- inspect a compact same-day evaluation snapshot if needed

---

## Repository Structure

```text
project/
├── dashboard/                   # Original Phase 1 Streamlit dashboard
│   ├── app.py
│   └── requirements.txt
│
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/
│   │   └── phase2/             # Featured and processed Phase 2 data
│   └── sample/                 # Optional sample data
│
├── deployment/                 # Final Phase 3 deployment app
│   ├── app.py
│   ├── inference.py
│   ├── utils.py
│   ├── requirements.txt
│   └── assets/
│
├── docs/                       # Figures and exported outputs
│
├── notebooks/
│   ├── phase1/                 # Phase 1 EDA notebooks
│   └── phase2/                 # Phase 2 modeling notebooks, reports, models
│
├── src/                        # Optional reusable source code
├── .gitignore
├── README.md
└── requirements.txt