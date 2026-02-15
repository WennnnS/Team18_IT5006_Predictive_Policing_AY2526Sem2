# Team18_IT5006_Predictive_Policing_AY2526Sem2

Predictive Policing Project – IT5006 (AY2025/26 Semester 2)

---

## 📌 Project Overview

This project explores predictive policing through exploratory data analysis (EDA) and interactive dashboard development based on Chicago crime data (2015–2024).

Phase 1 includes:

- Literature review
- Temporal pattern analysis
- Spatial distribution study
- Crime-type correlation analysis
- Interactive Streamlit dashboard

---

## 📊 Dataset

Primary Dataset:
Chicago Crimes (2015–2024 subset)

Fields used:
- Date
- Primary Type
- Latitude
- Longitude
- District (for spatial aggregation)

---

## 📁 Repository Structure

project/
│
├── dashboard/ # Streamlit application
├── notebooks/ # EDA notebooks
├── docs/ # Generated figures & outputs
└── README.md


---

## Running the Dashboard (Local)

Activate environment:

..venv\Scripts\Activate.ps1


Run Streamlit:



python -m streamlit run dashboard/app.py


---

## Live Application

Streamlit Cloud Deployment:  
[Insert your Streamlit URL here]

---

## Key EDA Findings

- Stable long-term crime volume with pandemic-related structural break  
- Strong summer seasonality (July–August peak)  
- Pronounced hourly concentration (afternoon–evening)  
- Significant spatial clustering (top districts ≈ 55% of total incidents)  
- Crime-type-specific seasonal heterogeneity  

---

## Team

Team 18  
IT5006 – Fundamentals of Data Analytics  
AY2025/26 Semester 2