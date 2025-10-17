# Machine Learning Task: Student Score Prediction
# Dataset: Student Performance Factors (Kaggle)
# Tools: Python, Pandas, Matplotlib, Scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("/home/afzaal-ur-rahman/Desktop/Student_score_prediction/StudentPerformanceFactors.csv")

print("Dataset loaded successfully.")
print(df.head())
print("\nDataset Information:")
print(df.info())

# Data Cleaning
print("\nChecking for missing values:")
print(df.isnull().sum())

# Remove duplicates and missing values
df = df.drop_duplicates()
df = df.dropna()

# Convert categorical variables into numerical format
df_encoded = pd.get_dummies(df, drop_first=True)
print("\nData cleaning completed successfully.")
print("Encoded DataFrame shape:", df_encoded.shape)

# Data Visualization
plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

if 'Study_Hours' in df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(x='Study_Hours', y='Exam_Score', data=df)
    plt.title("Study Hours vs Exam Score")
    plt.show()

# Split Dataset
X = df_encoded.drop('Exam_Score', axis=1)
y = df_encoded['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split completed.")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Model Training (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear Regression model trained successfully.")

# Model Testing and Prediction
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel evaluation results:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Prediction Visualization
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction Line')
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores")
plt.legend()
plt.show()

# Polynomial Regression (Bonus)
if 'Study_Hours' in df.columns:
    print("\nPerforming polynomial regression on Study Hours...")

    X_poly = df[['Study_Hours']]
    y_poly = df['Exam_Score']

    poly_features = PolynomialFeatures(degree=2)
    X_poly_transformed = poly_features.fit_transform(X_poly)

    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
        X_poly_transformed, y_poly, test_size=0.2, random_state=42
    )

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train_poly)
    y_poly_pred = poly_model.predict(X_test_poly)

    print("Polynomial Regression MSE:", mean_squared_error(y_test_poly, y_poly_pred))
    print("Polynomial Regression R²:", r2_score(y_test_poly, y_poly_pred))

    plt.figure(figsize=(7,5))
    plt.scatter(df['Study_Hours'], df['Exam_Score'], color='blue', label='Actual')
    plt.scatter(df['Study_Hours'], poly_model.predict(X_poly_transformed), color='red', label='Predicted')
    plt.title("Polynomial Regression Fit (Degree 2)")
    plt.xlabel("Study Hours")
    plt.ylabel("Exam Score")
    plt.legend()
    plt.show()
