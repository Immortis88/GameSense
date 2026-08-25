# ============================================================
# GameSense — 03_regression.py
# Linear Regression to predict review_ratio (continuous target)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Create plots folder if missing
os.makedirs('../plots', exist_ok=True)

# ------------------------------------------------------------
# 1. Load Processed Dataset
# ------------------------------------------------------------
print("Loading cleaned dataset...")
df = pd.read_csv('../data/processed/steam_cleaned.csv')
print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Define Features (X) and Target (y)
# ------------------------------------------------------------
feature_cols = [
    'Price', 'owners_numeric', 'Peak CCU', 'Required age',
    'Achievements', 'Recommendations',
    'Average playtime forever', 'Median playtime forever',
    'Metacritic score', 'is_free',
    'genre_indie', 'genre_action', 'genre_casual', 'genre_adventure',
    'genre_simulation', 'genre_strategy', 'genre_rpg',
    'genre_early_access', 'genre_free_to_play', 'genre_sports',
    'genre_racing', 'genre_massively_multiplayer'
]

# Use existing features
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].fillna(0)
y = df['review_ratio']

print("\nNumber of features:", len(feature_cols))

# ------------------------------------------------------------
# 3. Feature Scaling (StandardScaler)
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 4. Train-Test Split (80% train, 20% test)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ------------------------------------------------------------
# 5. Train Linear Regression Model
# ------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ------------------------------------------------------------
# 6. Model Evaluation Metrics
# ------------------------------------------------------------
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print("\n--- Linear Regression Performance ---")
print(f"Training R2 Score: {train_r2:.4f}")
print(f"Testing R2 Score:  {test_r2:.4f}")
print(f"Testing MAE:       {test_mae:.4f}")
print(f"Testing MSE:       {test_mse:.4f}")

# ------------------------------------------------------------
# 7. Print Feature Coefficients
# ------------------------------------------------------------
print("\nFeature Coefficients:")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print(coef_df.to_string(index=False))

# ------------------------------------------------------------
# 8. Plot 1: Actual vs Predicted Scatter Plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_test, alpha=0.3, color='steelblue', s=10)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Ideal Prediction')
plt.title('Actual vs Predicted Review Ratio (Linear Regression)')
plt.xlabel('Actual Review Ratio')
plt.ylabel('Predicted Review Ratio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../plots/06_regression_actual_vs_predicted.png', bbox_inches='tight')
plt.close()
print("\nSaved plot: 06_regression_actual_vs_predicted.png")

# ------------------------------------------------------------
# 9. Plot 2: Top Feature Coefficients
# ------------------------------------------------------------
top_coefs = coef_df.head(10)
plt.figure(figsize=(9, 5))
plt.barh(top_coefs['Feature'], top_coefs['Coefficient'], color='coral', edgecolor='black')
plt.title('Top 10 Positive Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/07_regression_coefficients.png', bbox_inches='tight')
plt.close()
print("Saved plot: 07_regression_coefficients.png")

# ------------------------------------------------------------
# 10. Plot 3: Residuals Plot
# ------------------------------------------------------------
residuals = y_test - y_pred_test
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_test, residuals, alpha=0.3, color='purple', s=10)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot (Actual - Predicted)')
plt.xlabel('Predicted Review Ratio')
plt.ylabel('Residual')
plt.grid(True, alpha=0.3)
plt.savefig('../plots/08_regression_residuals.png', bbox_inches='tight')
plt.close()
print("Saved plot: 08_regression_residuals.png")

print("\nLinear Regression Completed!")
