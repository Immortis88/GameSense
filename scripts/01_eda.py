# ============================================================
# GameSense — 01_eda.py
# Basic Exploratory Data Analysis on Steam Games Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves charts instead of opening windows
import matplotlib.pyplot as plt
import sys
import os

# Fix Windows console encoding for non-ASCII game names
sys.stdout.reconfigure(encoding='utf-8')

# Create plots folder
os.makedirs('../plots', exist_ok=True)

# ============================================================
# 1. LOAD & INSPECT
# ============================================================
df = pd.read_csv('../data/raw/games.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print("Shape:", df.shape)
print("\nColumns:")
print(list(df.columns))
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

# ============================================================
# 2. MISSING VALUES
# ============================================================
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# ============================================================
# 3. BASIC STATISTICS
# ============================================================
print("\n" + "=" * 50)
print("BASIC STATISTICS (Numeric Columns)")
print("=" * 50)
print(df[['Price', 'Positive', 'Negative', 'Peak CCU',
          'Achievements', 'Average playtime forever']].describe())

# ============================================================
# 4. PRICE DISTRIBUTION
# ============================================================
plt.figure(figsize=(10, 6))
prices = df['Price'][df['Price'] <= 60]  # cap at $60 to avoid outlier stretch
plt.hist(prices, bins=30, color='steelblue', edgecolor='black')
plt.title('Price Distribution (Capped at $60)')
plt.xlabel('Price (USD)')
plt.ylabel('Number of Games')
plt.grid(axis='y', alpha=0.3)
plt.savefig('../plots/01_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: plots/01_price_distribution.png")

# ============================================================
# 5. TOP 15 GENRES
# ============================================================
# Genres are comma-separated, e.g. "Action,Adventure,RPG"
all_genres = df['Genres'].dropna().str.split(',')
genre_list = [genre.strip() for genres in all_genres for genre in genres]
genre_counts = pd.Series(genre_list).value_counts().head(15)

plt.figure(figsize=(10, 6))
genre_counts.sort_values().plot(kind='barh', color='coral', edgecolor='black')
plt.title('Top 15 Genres on Steam')
plt.xlabel('Number of Games')
plt.ylabel('Genre')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/02_top_genres.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/02_top_genres.png")

# ============================================================
# 6. REVIEW RATIO DISTRIBUTION
# ============================================================
total_reviews = df['Positive'] + df['Negative']
# Only compute ratio where there are reviews
mask = total_reviews > 0
review_ratio = df.loc[mask, 'Positive'] / total_reviews[mask]

plt.figure(figsize=(10, 6))
plt.hist(review_ratio, bins=50, color='mediumseagreen', edgecolor='black')
plt.title('Review Ratio Distribution (Positive / Total)')
plt.xlabel('Review Ratio')
plt.ylabel('Number of Games')
plt.grid(axis='y', alpha=0.3)
plt.savefig('../plots/03_review_ratio.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/03_review_ratio.png")

# ============================================================
# 7. FREE vs PAID — Average Review Ratio
# ============================================================
df_with_reviews = df[mask].copy()
df_with_reviews['review_ratio'] = review_ratio
df_with_reviews['is_free'] = (df_with_reviews['Price'] == 0).map({True: 'Free', False: 'Paid'})

avg_ratio = df_with_reviews.groupby('is_free')['review_ratio'].mean()
counts = df_with_reviews.groupby('is_free')['review_ratio'].count()

print("\n" + "=" * 50)
print("FREE vs PAID")
print("=" * 50)
print("Count:\n", counts)
print("\nAvg Review Ratio:\n", avg_ratio)

plt.figure(figsize=(8, 5))
avg_ratio.plot(kind='bar', color=['#ff7f0e', '#1f77b4'], edgecolor='black')
plt.title('Average Review Ratio: Free vs Paid Games')
plt.xlabel('Game Type')
plt.ylabel('Average Review Ratio')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/04_free_vs_paid.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/04_free_vs_paid.png")

# ============================================================
# 8. GAMES RELEASED PER YEAR
# ============================================================
df['release_year'] = pd.to_datetime(df['Release date'], format='mixed', errors='coerce').dt.year
year_counts = df['release_year'].dropna().astype(int)
year_counts = year_counts[(year_counts >= 2000) & (year_counts <= 2025)]
year_counts = year_counts.value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(year_counts.index, year_counts.values, marker='o', color='purple', linewidth=2)
plt.title('Number of Games Released Per Year on Steam')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/05_games_per_year.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/05_games_per_year.png")

print("\n" + "=" * 50)
print("EDA COMPLETE — All charts saved to plots/ folder")
print("=" * 50)
