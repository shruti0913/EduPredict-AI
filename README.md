# EduPredict AI 🎓

## About the Project
EduPredict AI is a machine learning-based educational prototype designed to identify students who may be at risk of poor academic performance.

It provides an interactive dashboard, student risk prediction, and an Explainable AI visualization to help users understand model predictions.

## Features
- Interactive dashboard with student statistics
- Student academic risk prediction
- Explainable AI feature-contribution visualization
- Logistic Regression and Random Forest model comparison
- Data visualization using Plotly

## Dataset
This project uses the UCI Student Performance dataset, specifically `student-mat.csv`.

- Total records: 395
- Original dataset columns: 33
- Target: At Risk when final grade (G3) is below 10; otherwise Not At Risk
- G1, G2, and G3 are excluded from the model's input features.

## Model Performance

The selected model is Logistic Regression.

| Metric | Result |
|---|---:|
| Accuracy | 69.6% |
| At-risk Precision | 52.9% |
| At-risk Recall | 69.2% |
| At-risk F1-score | 0.600 |

Random Forest achieved 70.9% accuracy and an at-risk F1-score of 0.511.

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Project Structure

```text
EduPredict AI/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
└── src/
    └── train_model.py
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

### 3. Start the dashboard

```bash
streamlit run app.py
```

## Limitations
This project is an educational prototype, not a validated system for making real student-related decisions. Predictions may be inaccurate and should not be used as the sole basis for academic interventions.

## Future Improvements
- Evaluate the model on additional datasets
- Improve at-risk student recall
- Add more model evaluation visualizations
- Deploy the application online

## Author
Student Machine Learning Project