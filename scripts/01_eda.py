# ============================================================
# GameSense — 01_eda.py
# Basic Exploratory Data Analysis (EDA) on Steam Games Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots directory if it doesn't exist
os.makedirs('../plots', exist_ok=True)

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
print("Loading raw Steam games dataset...")
df = pd.read_csv('../data/raw/games.csv')

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Data Types:")
print(df.dtypes)

# ------------------------------------------------------------
# 2. Check Missing Values
# ------------------------------------------------------------
print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# ------------------------------------------------------------
# 3. Summary Statistics for Key Numeric Columns
# ------------------------------------------------------------
print("\nSummary Statistics:")
print(df[['Price', 'Positive', 'Negative', 'Achievements', 'Average playtime forever']].describe())

# ------------------------------------------------------------
# 4. Plot 1: Price Distribution
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
# Filtering prices <= $60 so outliers don't stretch graph
prices = df['Price'][df['Price'] <= 60]
plt.hist(prices, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Game Prices (up to $60)')
plt.xlabel('Price (USD)')
plt.ylabel('Number of Games')
plt.grid(True, alpha=0.3)
plt.savefig('../plots/01_price_distribution.png', bbox_inches='tight')
plt.close()
print("Saved plot: 01_price_distribution.png")

# ------------------------------------------------------------
# 5. Plot 2: Top 15 Genres
# ------------------------------------------------------------
# Split genres by comma and count occurrences
all_genres = df['Genres'].dropna().str.split(',')
genre_list = []
for genres in all_genres:
    for g in genres:
        genre_list.append(g.strip())

genre_counts = pd.Series(genre_list).value_counts().head(15)

plt.figure(figsize=(9, 5))
genre_counts.sort_values().plot(kind='barh', color='coral', edgecolor='black')
plt.title('Top 15 Most Common Genres on Steam')
plt.xlabel('Number of Games')
plt.ylabel('Genre')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/02_top_genres.png', bbox_inches='tight')
plt.close()
print("Saved plot: 02_top_genres.png")

# ------------------------------------------------------------
# 6. Plot 3: Review Ratio Distribution
# ------------------------------------------------------------
total_reviews = df['Positive'] + df['Negative']
# Only analyze games that have at least 1 review
has_reviews = total_reviews > 0
review_ratio = df.loc[has_reviews, 'Positive'] / total_reviews[has_reviews]

plt.figure(figsize=(8, 5))
plt.hist(review_ratio, bins=40, color='mediumseagreen', edgecolor='black')
plt.title('Distribution of Positive Review Ratios')
plt.xlabel('Review Ratio (Positive / Total)')
plt.ylabel('Number of Games')
plt.grid(True, alpha=0.3)
plt.savefig('../plots/03_review_ratio.png', bbox_inches='tight')
plt.close()
print("Saved plot: 03_review_ratio.png")

# ------------------------------------------------------------
# 7. Plot 4: Free vs Paid Games - Average Review Ratio
# ------------------------------------------------------------
df_reviews = df[has_reviews].copy()
df_reviews['review_ratio'] = review_ratio
df_reviews['is_free'] = df_reviews['Price'] == 0

avg_ratio = df_reviews.groupby('is_free')['review_ratio'].mean()

plt.figure(figsize=(6, 4))
plt.bar(['Paid', 'Free'], [avg_ratio[False], avg_ratio[True]], color=['#1f77b4', '#ff7f0e'], edgecolor='black')
plt.title('Average Review Ratio: Free vs Paid Games')
plt.ylabel('Average Review Ratio')
plt.grid(axis='y', alpha=0.3)
plt.savefig('../plots/04_free_vs_paid.png', bbox_inches='tight')
plt.close()
print("Saved plot: 04_free_vs_paid.png")

# ------------------------------------------------------------
# 8. Plot 5: Games Released Per Year
# ------------------------------------------------------------
df['release_year'] = pd.to_datetime(df['Release date'], errors='coerce').dt.year
year_counts = df['release_year'].value_counts().sort_index()
year_counts = year_counts[(year_counts.index >= 2000) & (year_counts.index <= 2025)]

plt.figure(figsize=(9, 5))
plt.plot(year_counts.index, year_counts.values, marker='o', color='purple')
plt.title('Number of Steam Games Released Per Year (2000 - 2025)')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.grid(True, alpha=0.3)
plt.savefig('../plots/05_games_per_year.png', bbox_inches='tight')
plt.close()
print("Saved plot: 05_games_per_year.png")

print("\nExploratory Data Analysis Complete!")
