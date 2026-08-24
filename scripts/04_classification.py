# ============================================================
# GameSense — 04_classification.py
# Classification: KNN, Decision Tree, Random Forest
# Predicting binary 'success' label
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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

print("Features:", len(feature_cols))
print("Class balance:")
print(y.value_counts())

# ============================================================
# 3. TRAIN-TEST SPLIT & SCALING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# 4a. KNN CLASSIFIER
# ============================================================
print("\n" + "=" * 50)
print("KNN CLASSIFIER")
print("=" * 50)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_knn):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_knn):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_knn):.4f}")

# Log some correct and wrong predictions
test_df = df.iloc[X_test.index][['Name']].copy()
test_df['actual'] = y_test.values
test_df['predicted'] = y_pred_knn

correct = test_df[test_df['actual'] == test_df['predicted']].head(5)
wrong = test_df[test_df['actual'] != test_df['predicted']].head(5)

print("\nCorrect Predictions (sample):")
print(correct[['Name', 'actual', 'predicted']].to_string(index=False))
print("\nWrong Predictions (sample):")
print(wrong[['Name', 'actual', 'predicted']].to_string(index=False))

# ============================================================
# 4b. DECISION TREE
# ============================================================
print("\n" + "=" * 50)
print("DECISION TREE")
print("=" * 50)

dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_dt):.4f}")

# Visualize the tree
plt.figure(figsize=(25, 12))
plot_tree(dt, feature_names=feature_cols, class_names=['Fail', 'Success'],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title('Decision Tree (max_depth=5, showing top 3 levels)')
plt.tight_layout()
plt.savefig('../plots/09_decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: plots/09_decision_tree.png")

# Extract top rules from the tree
importances_dt = pd.Series(dt.feature_importances_, index=feature_cols)
top_rules = importances_dt.sort_values(ascending=False).head(5)
print("\nTop 5 Important Features (Decision Tree):")
for feat, imp in top_rules.items():
    print(f"  {feat}: {imp:.4f}")

# ============================================================
# 4c. RANDOM FOREST — Vary n_estimators
# ============================================================
print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

n_trees_list = [10, 50, 100, 200]
rf_results = []

for n_trees in n_trees_list:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred_rf)
    prec = precision_score(y_test, y_pred_rf)
    rec = recall_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf)

    rf_results.append({
        'n_estimators': n_trees,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })
    print(f"  Trees={n_trees:3d} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

# Plot accuracy vs number of trees
rf_df = pd.DataFrame(rf_results)
plt.figure(figsize=(10, 6))
plt.plot(rf_df['n_estimators'], rf_df['accuracy'], marker='o', label='Accuracy', linewidth=2)
plt.plot(rf_df['n_estimators'], rf_df['f1'], marker='s', label='F1 Score', linewidth=2)
plt.title('Random Forest: Performance vs Number of Trees')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Score')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/10_rf_trees_vs_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: plots/10_rf_trees_vs_performance.png")

# Feature importances from best RF (200 trees)
best_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
best_rf.fit(X_train_scaled, y_train)
importances_rf = pd.Series(best_rf.feature_importances_, index=feature_cols)
top_features = importances_rf.sort_values(ascending=False).head(10)

print("\nTop 10 Feature Importances (Random Forest, 200 trees):")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.4f}")

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind='barh', color='forestgreen', edgecolor='black')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/11_rf_feature_importances.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/11_rf_feature_importances.png")

# ============================================================
# 5. MASTER COMPARISON TABLE
# ============================================================
print("\n" + "=" * 50)
print("CLASSIFICATION COMPARISON")
print("=" * 50)

comparison = pd.DataFrame({
    'Model': ['KNN (k=5)', 'Decision Tree (depth=5)', 'Random Forest (200 trees)'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, best_rf.predict(X_test_scaled))
    ],
    'Precision': [
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_dt),
        precision_score(y_test, best_rf.predict(X_test_scaled))
    ],
    'Recall': [
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, best_rf.predict(X_test_scaled))
    ],
    'F1': [
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_dt),
        f1_score(y_test, best_rf.predict(X_test_scaled))
    ]
})
print(comparison.to_string(index=False))

print("\n" + "=" * 50)
print("CLASSIFICATION COMPLETE")
print("=" * 50)
