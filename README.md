# Ozone Depletion Severity Predictor — Streamlit App

Deploys notebook's Softmax Regression (multinomial logistic regression)
model behind the UI which is designed, so every number on screen (accuracy,
confusion matrix, feature importance, probabilities) comes from the real
trained model — not the mockup's placeholder numbers.

## 🚀 Live Demo

🔗 **[Launch the Live App](https://ozone-depletion-severity-predictor.streamlit.app/)**

## Files

| File | Purpose |
|---|---|
| `ozone_layer_depletion_dataset.csv` | Your dataset |
| `train_model.py` | Reproduces the notebook's exact pipeline (imputation → label encoding → 80/20 stratified split → RobustScaler → Softmax Regression) and saves the model + scaler + encoders + metrics |
| `app.py` | The Streamlit app (Predict / Model Insights / About tabs) |
| `requirements.txt` | Python dependencies |
| `artifacts/` | Created by `train_model.py` — model, scaler, encoders, metrics.json, feature_importance.json |

## 1. Run locally in VS Code

Open this folder in VS Code, then in its integrated terminal:

```bash
# (recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# install dependencies
pip install -r requirements.txt

# train the model once — creates the artifacts/ folder app.py needs
python train_model.py

# launch the app
streamlit run app.py
```

Streamlit opens the app at `http://localhost:8501`. Every slider/select on
the **Predict** tab feeds straight into the real model; **Model Insights**
shows the actual test-set accuracy, confusion matrix, and coefficient-based
feature importance from this run of `train_model.py`.

If you retrain later (new data, tuned hyperparameters), just re-run
`python train_model.py` — the app picks up the new `artifacts/` files on
its next restart (`@st.cache_resource` caches them per server process).

## 2. Deploy to Streamlit Community Cloud (free)

1. Push this folder to a GitHub repo, including the `artifacts/` folder
   (commit it after running `train_model.py` locally, so the cloud app
   doesn't need to retrain on startup) — or add a one-line startup hook
   that runs `train_model.py` if `artifacts/` is missing (see note below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**.
3. Pick the repo/branch, set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and starts the app.

**Note:** if you'd rather not commit the binary `artifacts/*.joblib`
files, add this to the very top of `app.py`'s artifact-loading section
so the cloud instance trains on first boot:

```python
import os, subprocess
if not os.path.exists("artifacts/softmax_model.joblib"):
    subprocess.run(["python", "train_model.py"], check=True)
```

## 3. Other free hosts that work the same way

Any host that runs `pip install -r requirements.txt` then
`streamlit run app.py` works identically: Render, Railway, Hugging Face
Spaces (Streamlit SDK), or a VM/Docker container of your own.

## Notes on faithfulness to the notebook

- `train_model.py` mirrors your notebook step-for-step: same imputer
  strategies, same feature/target split, same `RobustScaler` fit only on
  train, same target mapping (`Low→1, Moderate→2, Severe→3`), same
  `LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)`.
- One bug fix versus the notebook: the notebook reused a single
  `LabelEncoder` across all 4 categorical columns, which silently
  overwrote its mapping each time. `train_model.py` fits one encoder per
  column and saves all four, so encoding new user input is reliable.
- Feature importance uses mean `|coefficient|` across the model's
  one-vs-rest rows — matches the "matches the 0.73–0.87 correlations"
  note in your notebook, computed for real rather than hard-coded.
