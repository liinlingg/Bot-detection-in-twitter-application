# Twitter Bot Detection (2025 Refresh)

End-to-end notebook project that detects automated Twitter accounts by blending exploratory data analysis, description-level NLP, and gradient-boosted tree models on structured metadata.

## Highlights

- Single Jupyter notebook (`Twitter_Bot_Detection.ipynb`) that walks from raw CSV to trained classifier and evaluation plots.
- Hybrid feature space: TF‑IDF embeddings of user descriptions + engineered account-level signals.
- Modeling stack couples a calibrated Linear SVM for text scoring with XGBoost for the final binary decision.
- Production-minded touches: language filtering, log transforms on heavy-tailed counts, class-imbalance handling, and feature-importance introspection.

## Dataset & Features

- Source: [Twitter Human Bots Dataset](https://huggingface.co/datasets/airt-ml/twitter-human-bots). Download `twitter_human_bots_dataset.csv` into `data/`.
- Target: `account_type` (`bot` vs `human`).
- Filtering: keep only English tweets (`lang == "en"`), drop redundant identifiers and image URLs, rename `default_profile` → `default_background_image`.
- Numeric features: follower/friend/favorite counts, tweets per day, account age.
- Boolean features: default imagery flags, geo enabled, verified.
- Text feature: profile `description`, vectorized with TF‑IDF (stop words removed, max 100 features for the current run).

## Repository Layout

```
.
├── Twitter_Bot_Detection.ipynb   # Complete workflow
├── data/
│   └── twitter_human_bots_dataset.csv
├── image/                        # Optional exported charts
└── requirements.txt              # Reproducible environment spec
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Python 3.9+ works; the notebook was last executed with Python 3.11. Dependencies include pandas, numpy, scipy, scikit-learn, seaborn, matplotlib, and xgboost (full list lives in `requirements.txt`).

## Running the Notebook

1. Place the dataset under `data/`.
2. Activate your virtual environment.
3. Launch Jupyter:
   ```bash
   jupyter notebook Twitter_Bot_Detection.ipynb
   ```
4. Run the cells sequentially. The notebook resets random seeds for reproducibility; feel free to re-run individual sections when experimenting.

### Notebook Roadmap

1. **EDA** – schema summary, missing-data audit, class distribution, histograms/boxplots, boolean odds ratios, TF‑IDF top terms per label.
2. **Feature Engineering** – log1p transforms on skewed counts, boolean consolidation, description statistics (length, hashtags, mentions), TF‑IDF vectorization.
3. **Modeling**  
   - Train `CalibratedClassifierCV(LinearSVC)` on text to obtain `P(bot | description)`.  
   - Append this probability to structured features and train an `XGBClassifier` with `scale_pos_weight` to address class imbalance.
4. **Evaluation** – classification report, confusion matrix, precision/recall curves, and XGBoost feature-importance plots.

## Typical Outputs

- Precision/recall/F1 per class plus macro/micro averages.
- Confusion matrix highlighting false bot alarms vs missed bots.
- Feature importances showing which account statistics or TF‑IDF signals drive decisions.
- Saved charts (optional) under `image/` for portfolio or presentation use.

## What I Focused On

- Keeping the exploratory phase and modeling logic in the same artifact for transparency.
- Demonstrating how to merge text and tabular signals without a heavyweight pipeline framework.
- Providing a reproducible environment via `requirements.txt` and deterministic seeds.

## Extending the Project

- Build time-based train/test splits to check temporal drift.
- Promote the notebook into a Python module or CLI for batch scoring.
- Track experiments with MLflow or Weights & Biases.
- Swap in transformers (e.g., `sentence-transformers`) for richer description embeddings.
- Automate linting/testing via pre-commit hooks or CI.

Contributions and suggestions are welcome! Open an issue or PR if you experiment with new features, datasets, or evaluation metrics.
