# ============================================================
# GameSense — 04_classification.py
# Classification Suite: KNN, Decision Tree, Random Forest
# Predicting binary 'success' label (1 = Success, 0 = Fail)
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
import os

# Create plots folder if missing
os.makedirs('../plots', exist_ok=True)

# ------------------------------------------------------------
# 1. Load Processed Dataset
# ------------------------------------------------------------
print("Loading processed dataset...")
df = pd.read_csv('../data/processed/steam_cleaned.csv')
print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Select Features (X) and Target (y)
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
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].fillna(0)
y = df['success']

print("Class distribution (0 = Fail, 1 = Success):")
print(y.value_counts())

# ------------------------------------------------------------
# 3. Train-Test Split & Feature Scaling
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# ------------------------------------------------------------
# 4. Model 1: k-Nearest Neighbors (KNN, k=5)
# ------------------------------------------------------------
print("\n" + "=" * 40)
print("1. KNN CLASSIFIER (k=5)")
print("=" * 40)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

print(f"Accuracy:  {acc_knn:.4f}")
print(f"Precision: {prec_knn:.4f}")
print(f"Recall:    {rec_knn:.4f}")
print(f"F1 Score:  {f1_knn:.4f}")

# Sample predictions
test_df = df.iloc[X_test.index][['Name']].copy()
test_df['actual'] = y_test.values
test_df['predicted'] = y_pred_knn

print("\nSample Correct Predictions:")
print(test_df[test_df['actual'] == test_df['predicted']].head(3)[['Name', 'actual', 'predicted']])

print("\nSample Wrong Predictions:")
print(test_df[test_df['actual'] != test_df['predicted']].head(3)[['Name', 'actual', 'predicted']])

# ------------------------------------------------------------
# 5. Model 2: Decision Tree (max_depth=5, Entropy)
# ------------------------------------------------------------
print("\n" + "=" * 40)
print("2. DECISION TREE (depth=5)")
print("=" * 40)

dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print(f"Accuracy:  {acc_dt:.4f}")
print(f"Precision: {prec_dt:.4f}")
print(f"Recall:    {rec_dt:.4f}")
print(f"F1 Score:  {f1_dt:.4f}")

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=feature_cols, class_names=['Fail', 'Success'],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title('Decision Tree Visualization (Top 3 Levels)')
plt.tight_layout()
plt.savefig('../plots/09_decision_tree.png', bbox_inches='tight')
plt.close()
print("Saved plot: 09_decision_tree.png")

# Top feature importances in Decision Tree
importances_dt = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 5 Important Features (Decision Tree):")
print(importances_dt.head(5))

# ------------------------------------------------------------
# 6. Model 3: Random Forest (Varying Number of Trees)
# ------------------------------------------------------------
print("\n" + "=" * 40)
print("3. RANDOM FOREST")
print("=" * 40)

tree_counts = [10, 50, 100, 200]
rf_accuracies = []
rf_f1s = []

for n_trees in tree_counts:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf)
    
    rf_accuracies.append(acc)
    rf_f1s.append(f1)
    print(f"Trees: {n_trees:3d} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

# Plot Accuracy vs Number of Trees
plt.figure(figsize=(8, 5))
plt.plot(tree_counts, rf_accuracies, marker='o', label='Accuracy', color='blue')
plt.plot(tree_counts, rf_f1s, marker='s', label='F1 Score', color='green')
plt.title('Random Forest Performance vs Number of Trees')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../plots/10_rf_trees_vs_performance.png', bbox_inches='tight')
plt.close()
print("Saved plot: 10_rf_trees_vs_performance.png")

# Best Random Forest (200 trees)
best_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
best_rf.fit(X_train_scaled, y_train)
y_pred_best_rf = best_rf.predict(X_test_scaled)

acc_rf = accuracy_score(y_test, y_pred_best_rf)
prec_rf = precision_score(y_test, y_pred_best_rf)
rec_rf = recall_score(y_test, y_pred_best_rf)
f1_rf = f1_score(y_test, y_pred_best_rf)

# Feature Importances Plot for Random Forest
importances_rf = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
plt.figure(figsize=(9, 5))
importances_rf.sort_values().plot(kind='barh', color='forestgreen', edgecolor='black')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/11_rf_feature_importances.png', bbox_inches='tight')
plt.close()
print("Saved plot: 11_rf_feature_importances.png")

# ------------------------------------------------------------
# 7. Summary Comparison Table
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("FINAL CLASSIFICATION MODEL COMPARISON")
print("=" * 50)

summary_df = pd.DataFrame({
    'Model': ['KNN (k=5)', 'Decision Tree (depth=5)', 'Random Forest (200 trees)'],
    'Accuracy': [acc_knn, acc_dt, acc_rf],
    'Precision': [prec_knn, prec_dt, prec_rf],
    'Recall': [rec_knn, rec_dt, rec_rf],
    'F1 Score': [f1_knn, f1_dt, f1_rf]
})

print(summary_df.to_string(index=False))
print("\nClassification Suite Complete!")
