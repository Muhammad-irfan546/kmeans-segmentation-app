# K-Means Customer/Product Segmentation App

A Streamlit app that loads a trained K-Means clustering model and predicts
cluster membership for new data.

## Files

- `app.py` — the Streamlit app
- `save_model_in_colab.py` — run this in Colab to export your trained model
- `kmeans_model.pkl` — your trained model (you generate this)
- `scaler.pkl` — your fitted StandardScaler (you generate this)
- `requirements.txt` — dependencies for Streamlit Cloud

## Step 1 — Save your model in Colab

At the bottom of your Colab notebook, after training, run the code in
`save_model_in_colab.py`. Adjust the variable names (`kmeans`, `scaler`)
to match your actual notebook. This will download two files to your
computer:

- `kmeans_model.pkl`
- `scaler.pkl`

## Step 2 — Edit app.py

Open `app.py` and edit this line near the top so it matches your actual
column names, in the same order you trained on:

```python
FEATURE_NAMES = ["Quantity", "UnitPrice", "TotalPrice"]
```

## Step 3 — Put all files in one folder

You should have these 4 files together:

```
kmeans-streamlit-app/
├── app.py
├── requirements.txt
├── kmeans_model.pkl
└── scaler.pkl
```

## Step 4 — Upload to GitHub

1. Go to https://github.com/Muhammad-irfan546
2. Click **New repository** → name it (e.g. `kmeans-segmentation-app`) → Create
3. Click **Add file → Upload files**
4. Drag in all 4 files above
5. Click **Commit changes**

Or via terminal:

```bash
git init
git add .
git commit -m "Add K-Means Streamlit app"
git branch -M main
git remote add origin https://github.com/Muhammad-irfan546/kmeans-segmentation-app.git
git push -u origin main
```

## Step 5 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select your repo, branch `main`, and set main file path to `app.py`
5. Click **Deploy**

Your app will be live at a URL like:
`https://kmeans-segmentation-app.streamlit.app`

## Notes

- `kmeans_model.pkl` and `scaler.pkl` must be committed to the repo (not
  in `.gitignore`) — the app needs them at runtime.
- If your dataset is large, only the trained model + scaler need to be
  uploaded, not the raw dataset.
