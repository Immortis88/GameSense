# ============================================================
# GameSense — 02_preprocessing.py
# Data Cleaning & Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. LOAD RAW DATA
# ============================================================
df = pd.read_csv('../data/raw/games.csv')
print("Original shape:", df.shape)

# ============================================================
# 2. DROP USELESS COLUMNS
# ============================================================
drop_cols = [
    'About the game', 'Reviews', 'Header image', 'Website',
    'Support url', 'Support email', 'Metacritic url',
    'Screenshots', 'Movies', 'Notes', 'Score rank',
    'Supported languages', 'Full audio languages'
]
df.drop(columns=drop_cols, inplace=True, errors='ignore')
print("After dropping useless columns:", df.shape)

# ============================================================
# 3. ESTIMATED OWNERS — already numeric in this dataset version
# ============================================================
# Rename for clarity
df.rename(columns={'Estimated owners': 'owners_numeric'}, inplace=True)
print("\nOwners sample:")
print(df['owners_numeric'].describe())

# ============================================================
# 4. DROP GAMES WITH ZERO REVIEWS (can't compute target)
# ============================================================
df['total_reviews'] = df['Positive'] + df['Negative']
before = len(df)
df = df[df['total_reviews'] > 0].copy()
print(f"\nDropped {before - len(df)} games with zero reviews")
print("Remaining:", len(df), "games")

# ============================================================
# 5. COMPUTE REVIEW RATIO
# ============================================================
df['review_ratio'] = df['Positive'] / df['total_reviews']
print("\nReview ratio stats:")
print(df['review_ratio'].describe())

# ============================================================
# 6. ADD IS_FREE FLAG
# ============================================================
df['is_free'] = (df['Price'] == 0).astype(int)
print("\nFree vs Paid count:")
print(df['is_free'].value_counts())

# ============================================================
# 7. DEFINE SUCCESS LABEL
# ============================================================
# Success = above-median review_ratio AND above-median owners
median_ratio = df['review_ratio'].median()
median_owners = df['owners_numeric'].median()

print(f"\nMedian review ratio: {median_ratio:.4f}")
print(f"Median owners: {median_owners:.0f}")

df['success'] = ((df['review_ratio'] > median_ratio) &
                 (df['owners_numeric'] > median_owners)).astype(int)

# ============================================================
# 8. CHECK CLASS BALANCE
# ============================================================
print("\n" + "=" * 50)
print("CLASS BALANCE")
print("=" * 50)
print(df['success'].value_counts())
print(f"\nSuccess rate: {df['success'].mean():.2%}")

# ============================================================
# 9. MULTI-HOT ENCODE TOP 15 GENRES
# ============================================================
# Split comma-separated genres and find top 15
all_genres = df['Genres'].dropna().str.split(',')
genre_list = [g.strip() for genres in all_genres for g in genres]
top_genres = pd.Series(genre_list).value_counts().head(15).index.tolist()

print("\nTop 15 Genres:", top_genres)

# Create one-hot columns for each top genre
for genre in top_genres:
    col_name = 'genre_' + genre.replace(' ', '_').lower()
    df[col_name] = df['Genres'].fillna('').apply(
        lambda x, g=genre: 1 if g in [item.strip() for item in x.split(',')] else 0
    )

# ============================================================
# 10. HANDLE REMAINING MISSING VALUES
# ============================================================
# Fill missing numeric columns with 0
fill_zero_cols = ['Price', 'Achievements', 'Average playtime forever',
                  'Median playtime forever', 'Peak CCU',
                  'Metacritic score', 'Recommendations',
                  'Average playtime two weeks', 'Median playtime two weeks']
for col in fill_zero_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print("\nRemaining missing values:")
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
print(remaining_missing if len(remaining_missing) > 0 else "None!")

# ============================================================
# 11. FINAL DATASET OVERVIEW
# ============================================================
print("\n" + "=" * 50)
print("FINAL PROCESSED DATASET")
print("=" * 50)
print("Shape:", df.shape)
print("\nColumns:")
print(list(df.columns))
print("\nFirst 5 rows (key columns):")
print(df[['Name', 'Price', 'owners_numeric', 'review_ratio', 'is_free', 'success']].head(10))

# ============================================================
# 12. SAVE PROCESSED DATA
# ============================================================
df.to_csv('../data/processed/steam_cleaned.csv', index=False)
print("\nSaved to data/processed/steam_cleaned.csv")
print("PREPROCESSING COMPLETE")
