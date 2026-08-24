# ============================================================
# GameSense — 02_preprocessing.py
# Data Cleaning & Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 1. Load Raw Dataset
# ------------------------------------------------------------
print("Loading raw dataset...")
df = pd.read_csv('../data/raw/games.csv')
print("Original shape:", df.shape)

# ------------------------------------------------------------
# 2. Drop Unnecessary Columns (URLs, descriptions, image links)
# ------------------------------------------------------------
drop_cols = [
    'About the game', 'Reviews', 'Header image', 'Website',
    'Support url', 'Support email', 'Metacritic url',
    'Screenshots', 'Movies', 'Notes', 'Score rank',
    'Supported languages', 'Full audio languages'
]
df = df.drop(columns=drop_cols, errors='ignore')
print("Shape after dropping text/URL columns:", df.shape)

# ------------------------------------------------------------
# 3. Rename Estimated Owners Column
# ------------------------------------------------------------
df = df.rename(columns={'Estimated owners': 'owners_numeric'})

# ------------------------------------------------------------
# 4. Remove Games with Zero Reviews
# ------------------------------------------------------------
df['total_reviews'] = df['Positive'] + df['Negative']
df = df[df['total_reviews'] > 0].copy()
print("Shape after dropping games with 0 reviews:", df.shape)

# ------------------------------------------------------------
# 5. Create Target & Feature Columns
# ------------------------------------------------------------
# Calculate positive review ratio (0 to 1)
df['review_ratio'] = df['Positive'] / df['total_reviews']

# Binary flag for free games
df['is_free'] = (df['Price'] == 0).astype(int)

# Define 'success' label: above-median review ratio AND above-median owners
median_ratio = df['review_ratio'].median()
median_owners = df['owners_numeric'].median()

print("\nMedian Review Ratio:", median_ratio)
print("Median Owners:", median_owners)

df['success'] = ((df['review_ratio'] > median_ratio) & 
                 (df['owners_numeric'] > median_owners)).astype(int)

print("\nClass distribution for 'success':")
print(df['success'].value_counts())
print("Success Percentage: {:.2f}%".format(df['success'].mean() * 100))

# ------------------------------------------------------------
# 6. One-Hot Encode Top 15 Genres
# ------------------------------------------------------------
# Find top 15 genres
all_genres = df['Genres'].dropna().str.split(',')
genre_list = []
for genres in all_genres:
    for g in genres:
        genre_list.append(g.strip())

top_genres = pd.Series(genre_list).value_counts().head(15).index.tolist()
print("\nTop 15 Genres:", top_genres)

# Create 0/1 column for each top genre
for g in top_genres:
    col_name = 'genre_' + g.replace(' ', '_').lower()
    df[col_name] = df['Genres'].fillna('').apply(lambda x: 1 if g in [item.strip() for item in x.split(',')] else 0)

# ------------------------------------------------------------
# 7. Fill Missing Values for Numeric Columns
# ------------------------------------------------------------
numeric_cols = [
    'Price', 'Achievements', 'Average playtime forever',
    'Median playtime forever', 'Peak CCU', 'Metacritic score',
    'Recommendations'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ------------------------------------------------------------
# 8. Save Processed Dataset
# ------------------------------------------------------------
os.makedirs('../data/processed', exist_ok=True)
output_path = '../data/processed/steam_cleaned.csv'
df.to_csv(output_path, index=False)
print("\nCleaned dataset saved successfully to:", output_path)
print("Final processed shape:", df.shape)
