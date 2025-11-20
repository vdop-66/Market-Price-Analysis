import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. DATA LOADING ---
print("Loading dataset...")
# We use pandas to read the CSV file into a 'DataFrame' (like an Excel sheet in code)
df = pd.read_csv('boston.csv')

# Quick check of the data
print("\n--- Data Snapshot (First 5 rows) ---")
print(df.head())

print("\n--- Data Statistics ---")
print(df.describe())

# --- 2. EXPLORATORY DATA ANALYSIS (EDA) & SELECTION ---
# We want to predict 'MEDV' (Median Value of owner-occupied homes in $1000s)
target = 'MEDV'

# Let's see which columns correlate most with Price
print(f"\n--- Correlation with {target} ---")
correlations = df.corr()[target].sort_values(ascending=False)
print(correlations)

# Based on standard analysis of this dataset, we select 3 key features for our website:
# 1. RM: Average number of rooms (High positive correlation)
# 2. LSTAT: % lower status of population (High negative correlation)
# 3. PTRATIO: Pupil-teacher ratio (Negative correlation)
selected_features = ['RM', 'LSTAT', 'PTRATIO']

print(f"\nSelected Features for Model: {selected_features}")

# --- 3. DATA PREPROCESSING ---
# Separate the inputs (X) and the answer key (y)
X = df[selected_features]
y = df[target]

# Split into Training (80%) and Testing (20%) sets
# We study on the training set, and take the exam on the testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. MODEL TRAINING ---
print("\nTraining Models...")

# Model A: Linear Regression (Simple, fits a straight line)
# Formula: Price = Intercept + (Weight1 * Rooms) + (Weight2 * LSTAT) ...
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model B: Random Forest (Complex, uses many decision trees)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- 5. MODEL EVALUATION ---
def evaluate_model(model, name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    accuracy = r2_score(y_test, predictions) * 100
    print(f"\n--- {name} Performance ---")
    print(f"Average Error: ${mae*1000:.2f}") # multiplied by 1000 because MEDV is in 1000s
    print(f"Accuracy (R2 Score): {accuracy:.2f}%")

evaluate_model(lr_model, "Linear Regression")
evaluate_model(rf_model, "Random Forest")

# --- 6. EXPORT LOGIC FOR WEBSITE ---
# We print these out so we can "hardcode" them into our Javascript website
# This allows the website to work without a complex Python backend server!
print("\n--- WEB DEPLOYMENT DATA ---")
print("Copy these values into your React Website code:")
print(f"Intercept (Base Price): {lr_model.intercept_:.4f}")
print(f"Coefficients (Weights):")
for feature, coef in zip(selected_features, lr_model.coef_):
    print(f"  {feature}: {coef:.4f}")