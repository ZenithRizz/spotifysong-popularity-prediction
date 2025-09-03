# 🎵 Spotify Song Popularity Prediction  

This project predicts the **popularity of songs** using audio features from the [Spotify Dataset (1921–2020, 600k tracks)](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks).  

We apply **machine learning models** (Linear Regression, Random Forest, XGBoost) to understand what makes a track popular and evaluate their performance.  

---

## 📂 Project Structure  

```

spotify-song-popularity-prediction/
│── data/                     # Dataset (ignored in git; use Kaggle API to download)
│   └── .gitkeep
│
│── notebooks/                 # Analysis & experiments
│   ├── 01\_EDA.py          # Exploratory Data Analysis
│   ├── 02\_Preprocessing.py # Cleaning & feature prep
│   └── 03\_Modeling.py      # Training & evaluation
│
│── src/                       # Reusable Python modules
│   ├── utils.py               # Data paths & helpers
│   ├── preprocessing.py       # Cleaning, splitting, scaling
│   └── modeling.py            # Training & evaluation functions
│
│── scripts/                   # Automation scripts
│   ├── download\_data.sh       # Kaggle dataset fetch (Linux/Mac)
│   └── download\_data.ps1      # Kaggle dataset fetch (Windows)
│
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
│── .gitignore                 # Ignore venv, data/, etc.

````

---

## ⚙️ Setup Instructions  

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/spotify-song-popularity-prediction.git
cd spotify-song-popularity-prediction
````

### 2. Create and activate a virtual environment

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset from Kaggle

1. Create a Kaggle API token (`kaggle.json` from your Kaggle account → Settings → Create API Token).

2. Place `kaggle.json` here:

   * **Linux/Mac** → `~/.kaggle/kaggle.json`
   * **Windows** → `%HOMEPATH%\.kaggle\kaggle.json`

3. Run script to download:

```bash
bash scripts/download_data.sh   # Linux/Mac
# or
powershell -ExecutionPolicy Bypass -File .\scripts\download_data.ps1   # Windows
```

This will place the dataset in `./data/`.

---

## 🧪 Workflow

1. **EDA (01\_EDA.py)**

   * Inspect dataset size, missing values, duplicates
   * Plot popularity distribution
   * Correlation heatmap of audio features
   * Explore feature-target relationships

2. **Preprocessing (02\_Preprocessing.py)**

   * Drop irrelevant columns (`track_name`, `artists`, etc.)
   * Handle duplicates and missing values
   * Train-test split & feature scaling

3. **Modeling (03\_Models.py)**

   * Baseline: Linear Regression
   * Advanced: Random Forest, XGBoost
   * Metrics: RMSE, MAE, R²
   * Feature importance visualization

---

## 📊 Results (to be updated with your findings)

* **Linear Regression**: RMSE = ..., R² = ...
* **Random Forest**: RMSE = ..., R² = ...
* **XGBoost**: RMSE = ..., R² = ...

Key insights:

* Energy, danceability, and loudness strongly correlate with popularity.
* Duration has weak/no impact.
* Random Forest/XGBoost outperform Linear Regression.

---

## Future Improvements

* Hyperparameter tuning (GridSearchCV / Optuna)
* Try LightGBM or CatBoost
* Explore deep learning (Neural Nets)
* Add artist-level features (average popularity per artist)

---

## Acknowledgements

* Dataset: [Spotify Dataset 1921–2020 (Kaggle)](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks)
* Tools: Python, pandas, scikit-learn, XGBoost, seaborn, matplotlib

```


Do you also want me to generate a **`.gitignore` and `requirements.txt`** for you so your repo is ready to push to GitHub?
```
