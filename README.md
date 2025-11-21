# Twitter Bot Detection

Detect automated Twitter accounts by combining exploratory analysis, text analytics, and machine learning models (TF‑IDF + Linear SVM + XGBoost). The workflow lives in the single notebook `Twitter_Bot_Detection.ipynb`.

## Dataset

- `data/twitter_human_bots_dataset.csv` – tweets and account level metadata with the binary target `account_type` (`bot` or `human`).
- The notebook filters to English tweets (`lang == "en"`), cleans redundant columns (IDs, image URLs, locations), and renames `default_profile` into `default_background_image` to avoid ambiguity.
- Numeric signals: follower/friend counts, engagement stats, account age, tweet velocity, etc.
- Boolean signals: default imagery, geo settings, verification flags.
- Text signal: account description (`description`) used for TF‑IDF features.

> The dataset is downloaded from [Twitter Human Bots Dataset](https://huggingface.co/datasets/airt-ml/twitter-human-bots) and place it under `data/`.

## Repository Layout

```
.
├── Twitter_Bot_Detection.ipynb   # Full workflow: EDA → modeling → evaluation
├── data/
│   └── twitter_human_bots_dataset.csv
└── image/                        # Plots exported from the notebook (optional)
```

## Requirements

- Python 3.9+ (tested on 3.11)
- pip
- Packages used in the notebook: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `xgboost`, `jupyter`

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost jupyter
```

## How to Run

1. Activate your environment (see above).
2. Launch Jupyter:
   ```bash
   jupyter notebook Twitter_Bot_Detection.ipynb
   ```
3. Execute cells sequentially. Key stages you will see:
   - **EDA** – schema summary, missing value scan, class balance plot, numeric/boolean/text profiling, TF‑IDF top terms by class.
   - **Feature Engineering** – log-transform skewed count features, derive description statistics, encode booleans, TF‑IDF for text.
   - **Modeling** – train a Linear SVM (wrapped in `CalibratedClassifierCV`) on description text; feed its probabilities plus numeric/boolean features into an XGBoost classifier with `scale_pos_weight` to balance classes.
   - **Evaluation** – classification report, confusion matrix, precision‑recall curve inputs, and XGBoost feature importance.

All figures display inline; re-run sections individually if you tweak feature lists or hyperparameters.

## Extending the Project

- Add a formal train/test split per time period to detect temporal drift.
- Promote the notebook pipeline to a Python module or script for batch scoring.
- Track experiments with MLflow or Weights & Biases for easier comparison.
- Package requirements into `requirements.txt` and/or Conda environment for reproducibility.

Contributions via issues or pull requests are welcome—share ideas for new features, additional datasets, or better evaluation metrics.
