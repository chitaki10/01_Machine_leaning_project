# Student Performance Prediction Web App

[![Python](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-2.0-orange)](https://flask.palletsprojects.com/) [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2-green)](https://scikit-learn.org/) 

A **full-stack machine learning project** that predicts a student’s math score based on demographic and academic features. This repository contains an end-to-end pipeline implemented in Python: from data ingestion, cleaning, and EDA, through model training and hyperparameter tuning, to deployment as a Flask web application. The trained model is exposed via a live API (see [Deployed App](https://studentperformance-t9km.onrender.com/predictdata)) for real-time predictions.

## Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Pipeline](#pipeline)  
- [Technologies](#technologies)  
- [Usage](#usage)  
- [API Endpoint (Live Demo)](#api-endpoint-live-demo)  
- [Results](#results)  
- [Project Structure](#project-structure)  
- [License](#license)  
- [Contact](#contact)  

## Features

- **End-to-end Pipeline:** Modular code for data ingestion, preprocessing, model training (with hyperparameter tuning), and evaluation.  
- **Data Cleaning & EDA:** Handles missing or inconsistent data and visualizes feature distributions (e.g., gender, ethnicity, scores) for insights.  
- **Feature Engineering:** Encodes categorical features (gender, race/ethnicity, education level, lunch, test prep) and scales numerical features (reading and writing scores).  
- **Model Training:** Explores multiple regression models, uses cross-validated hyperparameter search to find the best fit. The final model achieves strong predictive performance.  
- **Flask Web App:** A REST API for making predictions. The app accepts student features via JSON and returns the predicted math score. Deployed live on Render ([try it here](https://studentperformance-t9km.onrender.com/predictdata)).  
- **Logging & Reproducibility:** Comprehensive logging at each pipeline step ensures traceability. The preprocessing object and best model are saved for consistent inference.  

## Dataset

The project uses a publicly available “Students Performance in Exams” dataset (e.g., from Kaggle). Key features include: 

- **Demographics:** `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`.  
- **Academic Scores:** `reading_score`, `writing_score`.  
- **Target Variable:** `math_score` (to predict).  

This dataset provides a realistic basis for regression modeling of student performance, helping to demonstrate data cleaning and model training workflows.

## Pipeline

1. **Data Ingestion:** Load the dataset from CSV. Split into training and test sets (e.g., 80/20 split).  
2. **Preprocessing:**  
   - *Categorical Encoding:* Convert text categories to numeric codes or one-hot vectors (for features like gender, race, etc.).  
   - *Scaling:* Standardize numerical features (reading and writing scores) to have mean 0 and unit variance.  
   - *Pipeline Object:* Combine transformers into a single preprocessing pipeline (using scikit-learn’s `ColumnTransformer`/`Pipeline`).  

3. **Model Training:**  
   - Evaluate multiple regression algorithms (e.g., Linear Regression, Random Forest, Gradient Boosting).  
   - Perform hyperparameter tuning (e.g., via `GridSearchCV`) to optimize performance on validation data.  
   - Select the best model (Gradient Boosting Regressor in our case) based on R² score.  
   - Log training progress and scores at each step.  

4. **Model Persistence:** Save the trained preprocessing object and model to disk (e.g., using `joblib`), enabling the web app to load and use them for predictions.  

```plaintext
[LOG 07:26:42] Best R² score on test set: 0.8723 (Gradient Boosting Regressor)
```

## Technologies

- Python  
- pandas & NumPy  
- scikit-learn  
- Flask  
- Joblib / Pickle  
- Git  
- Render  

## Usage

### Clone the Repository

```bash
https://github.com/chitaki10/01_Machine_leaning_project](https://github.com/chitaki10/Student_Performance_Prediction_Web_App
cd student-performance-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python src/data_ingestion.py
python src/data_transformation.py
python src/model_training.py
```

### Run Flask App

```bash
export FLASK_APP=app.py
flask run
```

Server will start at:

```
http://127.0.0.1:5000/
```

---

## Live Demo

Deployed on Render:

https://studentperformance-t9km.onrender.com/predictdata

Example Python request:

```python
import requests

url = "https://studentperformance-t9km.onrender.com/predictdata"

data = {
    "gender": "male",
    "race_ethnicity": "group C",
    "parental_level_of_education": "some college",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 80,
    "writing_score": 78
}

response = requests.post(url, json=data)
print(response.json())
```

---

## Results

Best model: Gradient Boosting Regressor  

R² Score:

```
0.872289435850088
```

Logs:

```
[2026-02-18 07:26:42] Best score among models: 0.8723  
[2026-02-18 07:26:42] Best model: Gradient Boosting Regressor  
```

---

## Project Structure

```
student-performance-prediction/
│
├── application.py
├── README.md
├── requirements.txt
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── artifacts/
├── notebook/
├── logs/
└── templates/
```

---

## License

MIT License

---

## Contact

GitHub:https://github.com/chitaki10

Project Link:  
https://github.com/chitaki10/01_Machine_leaning_project](https://github.com/chitaki10/Student_Performance_Prediction_Web_App
