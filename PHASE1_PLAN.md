# GameSense — Phase 1 Plan (Basic EDA & Preprocessing)

## What's Done in This Iteration

### Scripts Created
1. **`scripts/01_eda.py`** — Exploratory Data Analysis
   - Load & inspect dataset (shape, columns, dtypes)
   - Missing values check
   - Basic statistics on numeric columns
   - 5 Charts: Price distribution, Top 15 Genres, Review Ratio, Free vs Paid, Games per Year

2. **`scripts/02_preprocessing.py`** — Data Cleaning & Feature Engineering
   - Drop 13 useless columns (descriptions, URLs, images, etc.)
   - Parse `Estimated owners` string ranges → numeric midpoint
   - Drop games with zero reviews
   - Compute `review_ratio` (Positive / Total)
   - Add `is_free` flag
   - Define binary `success` label (above-median ratio AND owners)
   - Multi-hot encode top 15 genres
   - Handle missing values
   - Save cleaned CSV to `data/processed/steam_cleaned.csv`

### Folder Structure
```
GameSense/
├── data/
│   ├── raw/games.csv          ← original dataset
│   └── processed/             ← cleaned output goes here
├── scripts/
│   ├── 01_eda.py
│   └── 02_preprocessing.py
├── plots/
├── report/
└── requirements.txt
```

## What's Next (Future Iterations)
- Phase 3: Regression (Linear Regression on review_ratio)
- Phase 4: Classification (KNN, NB, SVM, Decision Tree, Random Forest, Neural Net)
- Phase 5: K-Means Clustering + PCA
- Phase 6: Synthesis & comparison
- Stretch: Streamlit app, SHAP explanations

## How to Run
```bash
cd scripts
python 01_eda.py
python 02_preprocessing.py
```
