# ============================================================
# GameSense — 04_classification.py
# Classification Lab: KNN, Decision Tree, and Random Forest
# Goal: Predict if a game is Successful (1) or Fail (0)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv('../data/processed/steam_cleaned.csv')

# Select features for model training
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

# Make sure features exist in dataset
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].fillna(0)
y = df['success']

print("Features selected:", len(feature_cols))
print("Class counts (0 = Fail, 1 = Success):")
print(y.value_counts())

# ------------------------------------------------------------
# 2. Train-Test Split & Feature Scaling
# ------------------------------------------------------------
# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 3. Model 1: KNN (K-Nearest Neighbors)
# ------------------------------------------------------------
print("\n--- Model 1: KNN (k=5) ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

print("Accuracy: ", acc_knn)
print("Precision:", prec_knn)
print("Recall:   ", rec_knn)
print("F1 Score: ", f1_knn)

# ------------------------------------------------------------
# 4. Model 2: Decision Tree
# ------------------------------------------------------------
print("\n--- Model 2: Decision Tree (max_depth=5) ---")
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("Accuracy: ", acc_dt)
print("Precision:", prec_dt)
print("Recall:   ", rec_dt)
print("F1 Score: ", f1_dt)

# Plot decision tree
plt.figure(figsize=(16, 8))
plot_tree(dt, feature_names=feature_cols, class_names=['Fail', 'Success'], max_depth=3, filled=True)
plt.title("Decision Tree Visualization")
plt.savefig('../plots/09_decision_tree.png')
plt.close()

# ------------------------------------------------------------
# 5. Model 3: Random Forest (Testing 10, 50, 100, 200 trees)
# ------------------------------------------------------------
print("\n--- Model 3: Random Forest ---")

# Try 10 trees
rf10 = RandomForestClassifier(n_estimators=10, random_state=42)
rf10.fit(X_train_scaled, y_train)
pred10 = rf10.predict(X_test_scaled)
acc10 = accuracy_score(y_test, pred10)
f1_10 = f1_score(y_test, pred10)
print("10 trees  -> Accuracy:", acc10, "F1:", f1_10)

# Try 50 trees
rf50 = RandomForestClassifier(n_estimators=50, random_state=42)
rf50.fit(X_train_scaled, y_train)
pred50 = rf50.predict(X_test_scaled)
acc50 = accuracy_score(y_test, pred50)
f1_50 = f1_score(y_test, pred50)
print("50 trees  -> Accuracy:", acc50, "F1:", f1_50)

# Try 100 trees
rf100 = RandomForestClassifier(n_estimators=100, random_state=42)
rf100.fit(X_train_scaled, y_train)
pred100 = rf100.predict(X_test_scaled)
acc100 = accuracy_score(y_test, pred100)
f1_100 = f1_score(y_test, pred100)
print("100 trees -> Accuracy:", acc100, "F1:", f1_100)

# Try 200 trees
rf200 = RandomForestClassifier(n_estimators=200, random_state=42)
rf200.fit(X_train_scaled, y_train)
pred200 = rf200.predict(X_test_scaled)
acc200 = accuracy_score(y_test, pred200)
f1_200 = f1_score(y_test, pred200)
print("200 trees -> Accuracy:", acc200, "F1:", f1_200)

acc_rf = acc200
prec_rf = precision_score(y_test, pred200)
rec_rf = recall_score(y_test, pred200)
f1_rf = f1_200

# Plot accuracy vs number of trees
trees = [10, 50, 100, 200]
accuracies = [acc10, acc50, acc100, acc200]
f1_scores = [f1_10, f1_50, f1_100, f1_200]

plt.figure(figsize=(7, 4))
plt.plot(trees, accuracies, marker='o', label='Accuracy')
plt.plot(trees, f1_scores, marker='s', label='F1 Score')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.title('Random Forest Performance')
plt.legend()
plt.savefig('../plots/10_rf_trees_vs_performance.png')
plt.close()

# Feature importances for 200 trees
importances = pd.Series(rf200.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
importances.head(10).plot(kind='barh', color='green')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('../plots/11_rf_feature_importances.png')
plt.close()

# ------------------------------------------------------------
# 6. Summary Table
# ------------------------------------------------------------
print("\n=== Model Summary ===")
print("KNN (k=5) Accuracy:        ", acc_knn)
print("Decision Tree Accuracy:    ", acc_dt)
print("Random Forest Accuracy:    ", acc_rf)
