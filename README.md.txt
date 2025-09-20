# CO₂ Catalyst Property Predictor

A Streamlit web app to predict **Faradaic Efficiency, Overpotential, Selectivity, and Stability** of CO₂ electrocatalysts using machine learning.

## Features
- Upload CSV files of catalyst candidates
- Predict performance metrics (FE, Overpotential, etc.)
- Visualize results with SHAP, heatmaps, and bar charts
- Download full predictions as CSV

## Demo
![screenshot](assets/screenshot.png)

## Installation
```bash
git clone https://github.com/yourusername/co2-catalyst-predictor.git
cd co2-catalyst-predictor
pip install -r requirements.txt
streamlit run app.py
