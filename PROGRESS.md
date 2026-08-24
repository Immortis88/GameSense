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
*(upcoming)*

---

## Phase 3: Classification (KNN, NB, SVM, Decision Tree, Random Forest, Neural Net)
*(upcoming)*

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
