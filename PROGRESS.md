# GameSense — Progress Log

All progress tracked with timestamps. Updated as each phase is completed.

---

## Phase 1: Setup & EDA

### [2026-08-24 11:14 IST] — Project Structure Created
- Created folder structure: `data/raw/`, `data/processed/`, `scripts/`, `plots/`, `report/`
- Moved `games.csv` (401 MB, 125,855 rows × 39 columns) to `data/raw/`
- Created `requirements.txt` (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Created `PHASE1_PLAN.md`

### [2026-08-24 11:33 IST] — EDA Script Completed (`scripts/01_eda.py`)
- Loaded dataset: 125,855 games, 39 columns
- Inspected shape, dtypes, missing values
- Key finding: Most missing — `Movies` (100%), `Score rank` (99.9%), `Reviews` (90%)
- Key finding: Median price = $0 (majority free games), Average = $17.90
- Key finding: Paid games have slightly higher avg review ratio (78%) vs free (74.3%)
- Generated 5 charts saved to `plots/`:
  - `01_price_distribution.png`
  - `02_top_genres.png`
  - `03_review_ratio.png`
  - `04_free_vs_paid.png`
  - `05_games_per_year.png`

### [2026-08-24 11:38 IST] — Preprocessing Script Completed (`scripts/02_preprocessing.py`)
- Dropped 13 useless columns (descriptions, URLs, images) → 39 to 26 columns
- Dropped 42,899 games with zero reviews → 82,956 remaining
- Computed `review_ratio` = Positive / (Positive + Negative), mean = 0.758
- Added `is_free` flag: Free = 49,500 / Paid = 33,456
- Defined binary `success` label (above-median review_ratio AND owners)
- Class balance: success=0 → 72,126 (86.9%) / success=1 → 10,830 (13.1%) ⚠️ SKEWED
- Multi-hot encoded top 15 genres (Indie, Action, Casual, Adventure, etc.)
- Handled remaining missing values
- Saved cleaned dataset to `data/processed/steam_cleaned.csv` (82,956 rows × 45 columns)

---

## Phase 2: Regression

### [2026-08-24 13:41 IST] — Linear Regression Completed (`scripts/03_regression.py`)
- Target: `review_ratio` (continuous, 0–1)
- Features: 22 columns (Price, owners, playtime, achievements, Metacritic, genres, etc.)
- StandardScaler applied, 80/20 train-test split (66,364 / 16,592)
- **Results:**
  - Training R²: 0.0402 | Testing R²: 0.0418
  - Training MAE: 0.1786 | Testing MAE: 0.1798
  - Training MSE: 0.0545 | Testing MSE: 0.0551
- **Key finding:** R² is very low (~4%) — metadata alone is a weak predictor of review ratio. This is expected and makes a good discussion point.
- **Top positive coefficients:** Average playtime, Metacritic score, Indie genre
- **Top negative coefficients:** is_free (-0.071), Price (-0.060), Median playtime (-0.028)
- **Interesting insight:** Both `is_free` AND `Price` have negative coefficients — meaning neither extreme (free nor expensive) helps review ratio
- Generated 3 charts:
  - `06_regression_actual_vs_predicted.png`
  - `07_regression_coefficients.png`
  - `08_regression_residuals.png`

---

## Phase 3: Classification (KNN, Decision Tree, Random Forest)

### [2026-08-24 13:46 IST] — Classification Completed (`scripts/04_classification.py`)
- Target: binary `success` label (13% positive, 87% negative — imbalanced)
- Features: 22 columns, StandardScaler applied, 80/20 stratified split
- **Results:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN (k=5) | 87.75% | 57.43% | 23.92% | 33.77% |
| Decision Tree (depth=5) | 91.92% | 62.84% | 93.21% | 75.07% |
| Random Forest (200 trees) | 92.58% | 70.11% | 75.25% | 72.59% |

- **Best by F1:** Decision Tree (0.7507) — high recall catches most successful games
- **Best by Precision:** Random Forest (70.11%) — fewer false positives
- **KNN struggled** with recall (23.9%) — class imbalance hurts KNN the most
- **Decision Tree top feature:** `owners_numeric` (96% importance — dominates!)
- **Random Forest top features:** owners_numeric (39%), Recommendations (13%), playtime (7.7%)
- Random Forest: accuracy improves from 91.76% (10 trees) → 92.58% (200 trees)
- Generated 3 charts:
  - `09_decision_tree.png` — tree visualization
  - `10_rf_trees_vs_performance.png` — accuracy vs n_estimators
  - `11_rf_feature_importances.png` — RF feature importances

---

## Phase 4: Clustering & PCA
*(upcoming)*

---

## Phase 5: Synthesis & Final Report
*(upcoming)*

---

## Notes & Observations
- `Estimated owners` in this dataset version is already numeric (int64), NOT a string range — no parsing needed
- `Price` is also integer (in cents or whole units), not float
- Dataset downloaded date: 2026-08-24
- Dataset row count at download: 125,855 (excluding header)
- Heavy class imbalance (13% success) — must use F1/precision-recall in classification, not accuracy
