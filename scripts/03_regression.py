# ============================================================
# GameSense — 03_regression.py
# Linear Regression to predict review_ratio (continuous, 0-1)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs('../plots', exist_ok=True)

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================
df = pd.read_csv('../data/processed/steam_cleaned.csv')
print("Loaded shape:", df.shape)

# ============================================================
# 2. SELECT FEATURES & TARGET
# ============================================================
# Target: review_ratio (continuous 0-1)
# Features: numeric + genre columns
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

# Keep only columns that exist in the dataframe
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['review_ratio']

print("\nFeatures used:", feature_cols)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# 3. HANDLE MISSING VALUES IN FEATURES
# ============================================================
X = X.fillna(0)
print("\nMissing values in X after fill:", X.isnull().sum().sum())

# ============================================================
# 4. FEATURE SCALING (StandardScaler)
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaling done (StandardScaler)")

# ============================================================
# 5. TRAIN-TEST SPLIT (80/20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size:  {X_test.shape[0]}")

# ============================================================
# 6. FIT LINEAR REGRESSION MODEL
# ============================================================
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("\nModel trained!")

# ============================================================
# 7. PREDICTIONS
# ============================================================
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# ============================================================
# 8. MODEL EVALUATION
# ============================================================
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Training MSE:  {train_mse:.6f}")
print(f"Testing MSE:   {test_mse:.6f}")
print(f"Training MAE:  {train_mae:.6f}")
print(f"Testing MAE:   {test_mae:.6f}")
print(f"Training R²:   {train_r2:.6f}")
print(f"Testing R²:    {test_r2:.6f}")

# ============================================================
# 9. MODEL INFORMATION
# ============================================================
print("\n" + "=" * 50)
print("MODEL INFORMATION")
print("=" * 50)
print(f"Intercept: {regressor.intercept_:.6f}")
print("\nFeature Coefficients:")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': regressor.coef_
}).sort_values('Coefficient', ascending=False)
print(coef_df.to_string(index=False))

# ============================================================
# 10. PLOT — ACTUAL vs PREDICTED (Test Data)
# ============================================================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.3, color='steelblue', s=10)
plt.plot([0, 1], [0, 1], color='red', linewidth=2, linestyle='--', label='Perfect Prediction')
plt.title('Linear Regression: Actual vs Predicted Review Ratio')
plt.xlabel('Actual Review Ratio')
plt.ylabel('Predicted Review Ratio')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/06_regression_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: plots/06_regression_actual_vs_predicted.png")

# ============================================================
# 11. PLOT — FEATURE COEFFICIENTS (Top 10)
# ============================================================
top_coefs = coef_df.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_coefs['Feature'], top_coefs['Coefficient'], color='coral', edgecolor='black')
plt.title('Top 10 Feature Coefficients (Linear Regression)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/07_regression_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/07_regression_coefficients.png")

# ============================================================
# 12. PLOT — RESIDUALS
# ============================================================
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.3, color='purple', s=10)
plt.axhline(y=0, color='red', linewidth=2, linestyle='--')
plt.title('Residual Plot (Test Data)')
plt.xlabel('Predicted Review Ratio')
plt.ylabel('Residual (Actual - Predicted)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/08_regression_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/08_regression_residuals.png")

print("\n" + "=" * 50)
print("REGRESSION COMPLETE")
print("=" * 50)
