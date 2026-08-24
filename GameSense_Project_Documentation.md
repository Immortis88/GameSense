# GameSense: Predicting Game Success and Discovering Genre Archetypes on Steam

## 1. Project Summary

**Research question:** What factors predict whether a Steam game succeeds, and do natural "genre archetypes" (discovered without labels) actually align with real-world success?

This project uses one dataset to walk through the full ML Lab syllabus (regression, classification, ensemble methods, neural nets, clustering, dimensionality reduction) as a single coherent pipeline rather than 11 disconnected scripts. Supervised models answer "does X predict success?" and unsupervised methods answer "do games cluster into natural types, and which types tend to succeed?" — then Phase 6 ties both answers together.

---

## 2. Dataset

**Source:** Kaggle
- Primary option: `fronkongames/steam-games-dataset` (frequently updated, large, clean-ish)
- Alternative: `nikdavis/steam-store-games` (older but well-documented, has separate genre/tag/description CSVs)

**Expected columns to look for:**
- `name`, `release_date`, `price`
- `genres`, `tags`/`steamspy_tags`, `categories`
- `positive_ratings`, `negative_ratings`
- `owners` (bucketed estimate, e.g. "20000-50000")
- `average_playtime`, `median_playtime`
- `achievements`, `platforms`, `developer`, `publisher`

**Download steps:**
1. Go to kaggle.com, search the dataset name above
2. Download the CSV (or use `kaggle datasets download -d fronkongames/steam-games-dataset` via Kaggle API/CLI if configured)
3. Place in `data/raw/steam_games.csv`

---

## 3. Target Variable Definition

Since "success" isn't a column, define it explicitly and document your reasoning in the report:

```python
df['review_ratio'] = df['positive_ratings'] / (df['positive_ratings'] + df['negative_ratings'])

# Success = above-median review ratio AND above-median owners estimate
df['success'] = ((df['review_ratio'] > df['review_ratio'].median()) &
                  (df['owners_numeric'] > df['owners_numeric'].median())).astype(int)
```

Note: `owners` is usually a string range like "0-20000" — convert to a numeric midpoint first. Document this transformation clearly since it's a judgment call an interviewer may ask about.

---

## 4. Environment Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install pandas numpy scikit-learn matplotlib seaborn jupyter shap streamlit
```

Recommended: Jupyter notebooks per phase, numbered, so the narrative is easy to follow when demoed.

---

## 5. Project Phases & Steps

### Phase 1 — Setup & Exploration
- Load dataset, check `.shape`, `.info()`, `.isnull().sum()`
- Define `success` label (see Section 3)
- EDA:
  - Distribution of genres (bar chart of top 15)
  - Price vs review_ratio scatter
  - Free-to-play vs paid success rate (bar chart)
  - Release year trend — has success rate changed over time?
- Tools: Pandas, Matplotlib, Seaborn

### Phase 2 — Preprocessing
- Parse `genres`/`tags` (semicolon or comma separated) → multi-hot encode top N (e.g. top 20 genres/tags, bucket rest as "other")
- Handle missing price/description values
- Convert `owners` string ranges to numeric midpoints
- Scale numeric features (`price`, `average_playtime`, `achievements`) with `StandardScaler`
- Stratified train-test split on `success`

### Phase 3 — Regression
- Target: `review_ratio` (continuous, 0–1) instead of binary success
- Model: Linear Regression (baseline), optionally Ridge/Lasso for comparison
- Evaluate: R², MAE; plot actual vs predicted scatter

### Phase 4 — Classification Suite
| Step | Model | What to report |
|---|---|---|
| 4a | KNN | Accuracy, log a few correct/wrong predictions with game names |
| 4b | Naive Bayes | Accuracy, compare to KNN |
| 4c | SVM (Linear, Polynomial, RBF) | Accuracy table across 3 kernels |
| 4d | Decision Tree (entropy/ID3) | Visualize tree (`plot_tree`), extract 3-5 human-readable rules |
| 4e | Random Forest | Vary `n_estimators` = [10, 50, 100, 200], plot accuracy vs trees, extract `feature_importances_` |
| 4f | Neural Network (MLPClassifier or Keras) | Same architecture, compare `sigmoid`/`tanh`/`relu`, plot loss curves |

Use **F1-score and precision/recall**, not just accuracy — success labels may be imbalanced depending on your threshold choice.

### Phase 5 — Unsupervised Layer (the differentiator)
- **K-Means**: cluster games on genre/tag/price/playtime vectors only (drop the success label)
  - Try k = 3 to 8
  - Use elbow method (inertia) + silhouette score to pick k
- **PCA**: reduce feature space to 2D for visualization
  - Plot clusters in PCA space, colored by cluster
  - Overlay actual success rate per cluster (color intensity or separate bar chart)
- **Name the clusters** based on dominant genres/tags/price range (e.g. "Budget Indie Puzzle", "AAA Open-World Action", "Free-to-Play Shooter") — this is your standout narrative piece for interviews

### Phase 6 — Capstone Synthesis
- Build one master comparison table: model | accuracy | precision | recall | F1
- Cross-reference: which cluster (archetype) has highest actual success rate? Does this match what Random Forest's feature importances suggested?
- Write a short "Top 5 factors predicting Steam game success" section
- **Optional stretch goal:** Streamlit app — user inputs genre/price/tags → get predicted success probability + nearest archetype cluster
- **Optional stretch goal:** SHAP values on the Random Forest to explain individual predictions ("this game is predicted successful because of low price + action+indie tags + high playtime")

---

## 6. Suggested Repo Structure

```
GameSense/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_regression.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_clustering_pca.ipynb
│   └── 06_synthesis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
├── app/
│   └── streamlit_app.py        # optional
├── report/
│   └── GameSense_Report.md
├── requirements.txt
└── README.md
```

---

## 7. Evaluation Checklist (map back to syllabus)

- [ ] Exp 1 — NumPy/Pandas/Matplotlib/scikit-learn for data handling → Phase 1
- [ ] Exp 2 — Cleaning, merging, scaling, encoding → Phase 2
- [ ] Exp 3 — Regression, plot graph → Phase 3
- [ ] Exp 4 — KNN + Naive Bayes, correct/wrong predictions → Phase 4a-b
- [ ] Exp 5 — SVM kernel comparison → Phase 4c
- [ ] Exp 6 — Decision Tree (ID3) → Phase 4d
- [ ] Exp 7 — Random Forest, vary trees → Phase 4e
- [ ] Exp 8 — K-Means, vary k → Phase 5
- [ ] Exp 9 — PCA → Phase 5
- [ ] Exp 10 — Neural net, activation comparison → Phase 4f
- [ ] Exp 11 — Mini project/case study → Phase 6 (this whole project IS the case study)

---

## 8. Interview Talking Points (prep these in advance)

1. **Why this dataset/domain?** — personal interest in gaming, wanted a less-saturated alternative to churn/titanic datasets
2. **Why this success definition?** — be ready to defend the threshold choice, and mention it's a limitation/assumption
3. **Biggest surprise in EDA** — have a real answer, e.g. "price had weaker correlation with success than I expected"
4. **Supervised vs unsupervised agreement/disagreement** — did the best-performing cluster archetype match what Random Forest said mattered? If not, why might that be?
5. **What you'd do with more time** — e.g. NLP on game descriptions, time-series of review velocity, matching your ongoing sentiment analysis internship work

---

## 9. Notes / Open Decisions to Fill In As You Go

- Final chosen dataset version and download date:
- Final N for top genres/tags encoding:
- Final success threshold definition (after checking class balance):
- Best k chosen for K-Means and why:
- Best performing classifier overall:
