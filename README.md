# Symptom-Based Disease Prediction using Machine Learning

This project implements a machine learning system to predict diseases based on patient symptoms using a Bernoulli Naive Bayes classifier.

## Overview

The notebook processes a medical dataset containing symptoms as input features and diseases as the target label.  
To improve model performance and class balance, only the top 50 most frequent diseases are considered.  
The trained model predicts disease probabilities and evaluates performance using Top-5 accuracy.

## Technology Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Dataset Description

- Each row represents a patient record
- Features are binary symptom indicators
- Target column: `diseases`
- The dataset also includes a `depression` indicator

Only the top 50 most frequent diseases are used for training and evaluation.

## Workflow

1. Load and inspect the dataset
2. Perform exploratory data analysis
3. Filter the top 50 disease classes
4. Encode disease labels using LabelEncoder
5. Split data into training and testing sets
6. Train a Bernoulli Naive Bayes classifier
7. Evaluate model performance using Top-5 accuracy
8. Save trained model and supporting files

## Machine Learning Model

- **Algorithm:** Bernoulli Naive Bayes  
- **Reason:** Suitable for binary symptom-based features

## Evaluation Metric

- **Top-5 Accuracy**

Top-5 accuracy measures whether the correct disease appears in the top 5 predicted probabilities for each sample.

## Model Saving

The following files are saved using Joblib:
- Trained Bernoulli Naive Bayes model
- Label encoder for disease labels
- List of symptom features

These files can be reused for deployment or inference.

## Requirements

- Python 3.x
- Jupyter Notebook

Install required libraries:

```bash
pip install pandas numpy scikit-learn joblib
```
## How to Run

1. Clone the repository
2. Open the project directory
3. Launch Jupyter Notebook
4. Open Code.ipynb
5. Run all cells sequentially

## Output

- Disease probability predictions
- Top-5 predicted diseases for a given symptom set
- Serialized model and preprocessing files

## Limitations

- Only top 50 diseases are considered
- Model relies on binary symptom presence
- Not intended for real clinical diagnosis

## Future Improvements

- Include all disease classes
- Use advanced classification models
- Add symptom severity weighting
- Deploy as a web application using Streamlit or Flask

## Author

Rakshith S
