# 🏏 IPL Live Win Predictor

A machine learning web application that predicts win probability during the second innings of an IPL match using **Logistic Regression** trained on 15+ seasons of IPL ball-by-ball data.

Built as part of MCA Data Science coursework at MIT World Peace University, Pune.

---

## 🎯 Problem Statement

Given the live state of a T20 cricket match during a chase (target, current score, wickets fallen, overs bowled), predict the probability that the batting team will win.

This mirrors the "Win Probability Meter" seen on live broadcasts like Star Sports and JioCinema.

---

## 📊 Dataset

- **Source:** [IPL Complete Dataset 2008-2024 (Kaggle)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- **Files used:**
  - `matches.csv` — 1,095 matches with results
  - `deliveries.csv` — 260,920 ball-by-ball records

---

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas, numpy** — data manipulation
- **scikit-learn** — Logistic Regression, preprocessing, evaluation
- **matplotlib, seaborn** — visualizations
- **Streamlit** — web application
- **joblib** — model serialization

---

## 🔬 Methodology

### 1. Data Preparation
- Merged `matches.csv` with `deliveries.csv` on `match_id`
- Filtered only second-innings balls (prediction makes sense during chase)
- Handled team name changes (Delhi Daredevils → Delhi Capitals, etc.)

### 2. Feature Engineering
Derived **6 contextual features** from raw ball-by-ball data:

| Feature | Description |
|---------|-------------|
| `current_score` | Cumulative runs scored in the chase |
| `runs_left` | Target − current score |
| `balls_left` | 120 − balls bowled |
| `wickets_left` | 10 − wickets fallen |
| `crr` | Current run rate (runs per over so far) |
| `rrr` | Required run rate to win |

### 3. Model Training
- **Algorithm:** Logistic Regression
- **Train/Test split:** 80/20 with stratification
- **Feature scaling:** StandardScaler
- **Target:** Binary — did batting team win? (1/0)

### 4. Evaluation

| Metric | Score |
|--------|-------|
| Accuracy | **77.53%** |
| Precision | 77.38% |
| Recall | 80.68% |
| F1 Score | 79.00% |

---
## 🚀 How to Run

### Step 1 — Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 2 — Train the model
\`\`\`bash
cd notebooks
jupyter notebook ipl_live_predictor.ipynb
\`\`\`
Run all cells. This generates `.pkl` files in the `models/` folder.

### Step 3 — Launch the web app
\`\`\`bash
cd webapp
python -m streamlit run app.py
\`\`\`

Open `http://localhost:8501` in your browser.
---


```

---

## ⚠️ Limitations

1. **Works only in the second innings** — during chase, when a concrete target exists. Mirrors real-world broadcast practice.
2. **No player-level data** — doesn't account for individual batsman form, bowler matchups, or injury status.
3. **No pitch/weather conditions** — ignores dew factor, overcast conditions.
4. **Historical bias** — trained on data up to 2024; newer franchises have fewer samples.

---

## 💡 Key Learnings

- **Feature engineering matters more than algorithm choice.** A simple logistic regression with well-engineered features outperforms naive approaches with raw data.
- **Data merging is half the battle.** Joining two CSVs on a common key enabled labeled training data at ball-level granularity.
- **Scale mismatch breaks deployment.** Saving the scaler along with the model is critical — without it, new inputs are transformed differently than training data.

---

## 👤 Author

**Smit Nikumbh**

---

## 📝 License

This project is for academic purposes as part of MCA Data Science coursework.
