---
title: Smart Grid IoT Analytics
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Smart Grid Energy Analytics
**Short Description:** IoT dashboard for energy forecasting and tamper detection.

This project implements an end-to-end machine learning pipeline to analyze smart meter data.
* **Demand Forecasting:** Predicts grid load using XGBoost Regression.
* **Theft Detection:** Identifies current bypass tampering using XGBoost Classification.
* **Explainability:** Utilizes SHAP values to interpret model decisions.

## Resources
* **Hugging Face Space:** [singhh-piyush/smart-grid-iot](https://huggingface.co/spaces/singhh-piyush/smart-grid-iot)
* **Dataset:** [IoT-Enabled Smart Grid Dataset (Kaggle)](https://www.kaggle.com/datasets/ziya07/iot-enabled-smart-grid-dataset)
