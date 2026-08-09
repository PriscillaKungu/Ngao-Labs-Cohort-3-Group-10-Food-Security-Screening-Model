import pickle

artifact_bundle = {
    "model": final_model,
    "label_encoder": le,
    "kept_cols": kept_cols,
    "dropped_cols": dropped_cols,
    "cat_cols": cat_cols,
    "bin_cols": bin_cols,
    "num_cols": num_cols,
    "asset_cols": asset_cols,
    "exp_cols": exp_cols,
    "transfer_value_cols": transfer_value_cols,
    "s2_q21_noadults_median": s2_q21_noadults_median,
    "numeric_medians": numeric_medians,
    "county_rate_map": county_rate_map,
    "global_insecurity_rate": global_insecurity_rate,
    "training_feature_columns": training_feature_columns,
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifact_bundle, f)

Run that cell, download model_artifacts.pkl, and put it in app/ in your repo.

A note on model choice for deployment: if your final model ended up being the weighted-average ensemble or the stacking model (rather than a single tuned XGBoost), bundle each fitted sub-model plus the blend weights instead of one final_model, and adjust the /predict handler in Step 2 to combine their predict_proba outputs the same way the notebook does. A single tuned XGBoost model — which is what won in this project — is simplest to deploy and is the version this manual assumes.

File size check: GitHub blocks files over 100 MB (warns above ~50 MB). Tree models like these are usually a few MB, so you're almost certainly fine. If it's large, use Git LFS (git lfs track "*.pkl") before committing.

Step 2 — Build the web app

Create app/main.py. This wraps preprocess_new_household() from your notebook and exposes both a single-household endpoint and a batch CSV endpoint:

python
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import io

app = FastAPI(title="Household Food Security Screening")
templates = Jinja2Templates(directory="templates")

with open("model_artifacts.pkl", "rb") as f:
    A = pickle.load(f)

model = A["model"]
le = A["label_encoder"]

def preprocess_new_household(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Same logic as the notebook's preprocess_new_household(), reading
    everything it needs from the loaded artifact bundle A."""
    processed_df = raw_data.copy()

    for col in A["asset_cols"]:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df["n_assets_owned"] = processed_df[A["asset_cols"]].sum(axis=1)

    processed_df["is_farming_hh"] = processed_df["s4_q4_acres"].notna().astype(int)
    processed_df["acres_if_farm"] = processed_df["s4_q4_acres"].fillna(0)
    processed_df["livestock_value_filled"] = processed_df["s4_q9_livestockvalue"].fillna(0)
    processed_df["num_enterprises"] = processed_df["s4_q12_num_ent"].fillna(0)

    for c in A["transfer_value_cols"]:
        processed_df[c + "_f"] = processed_df[c].fillna(0)
    processed_df["total_transfer_value"] = processed_df[[c + "_f" for c in A["transfer_value_cols"]]].sum(axis=1)

    for col in A["exp_cols"]:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df["total_expenditure"] = processed_df[A["exp_cols"]].sum(axis=1)
    processed_df["food_expenditure_share"] = np.where(
        processed_df["total_expenditure"] > 0,
        processed_df["s5_q3a_food"].fillna(0) / processed_df["total_expenditure"],
        np.nan,
    )
    processed_df["food_expenditure_share"] = processed_df["food_expenditure_share"].fillna(
        A["numeric_medians"]["food_expenditure_share"]
    )

    processed_df["s2_q21_noadults_f"] = processed_df["s2_q21_noadults"].fillna(A["s2_q21_noadults_median"])
    processed_df["dependency_ratio"] = (
        (processed_df["s2_q15b_schoolchildren"].fillna(0) + processed_df["s2_q15c_youngchildren"].fillna(0))
        / processed_df["s2_q21_noadults_f"].replace(0, 1)
    )

    cols_to_drop = [c for c in A["dropped_cols"] if c in processed_df.columns]
    processed_df = processed_df.drop(columns=cols_to_drop, errors="ignore")

    current_num = [c for c in A["num_cols"] if c in A["kept_cols"]]
    current_bin = [c for c in A["bin_cols"] if c in A["kept_cols"]]
    current_cat = [c for c in A["cat_cols"] if c in A["kept_cols"]]

    for c in current_num:
        if c in processed_df.columns and processed_df[c].isnull().any():
            processed_df[c] = processed_df[c].fillna(A["numeric_medians"][c])
        elif c not in processed_df.columns:
            processed_df[c] = A["numeric_medians"][c]
    for c in current_bin:
        if c in processed_df.columns and processed_df[c].isnull().any():
            processed_df[c] = processed_df[c].fillna(0)
        elif c not in processed_df.columns:
            processed_df[c] = 0

    for c in current_cat:
        processed_df[c] = processed_df[c].fillna("missing").astype(str) if c in processed_df.columns else "missing"
    X_cat = pd.get_dummies(processed_df[current_cat], prefix=current_cat, dtype=int)

    county_series = processed_df["s2_q9a_county"].fillna("missing").astype(str)
    processed_df["county_insecurity_rate"] = county_series.map(A["county_rate_map"]).fillna(A["global_insecurity_rate"])

    final_df = pd.concat([
        processed_df[current_bin + current_num].reset_index(drop=True),
        X_cat.reset_index(drop=True),
        processed_df[["county_insecurity_rate"]].reset_index(drop=True),
    ], axis=1)

    return final_df.reindex(columns=A["training_feature_columns"], fill_value=0)


def score_households(raw_df: pd.DataFrame) -> pd.DataFrame:
    features = preprocess_new_household(raw_df)
    proba = model.predict_proba(features)
    pred_idx = np.argmax(proba, axis=1)
    labels = le.inverse_transform(pred_idx)

    # Order categories from most to least urgent for ranking purposes.
    urgency_rank = {"severely_insecure": 0, "moderately_insecure": 1,
                     "mildly_insecure": 2, "food_secure": 3}

    result = raw_df.copy()
    result["predicted_status"] = labels
    result["confidence"] = proba[np.arange(len(proba)), pred_idx]
    result["priority_rank"] = result["predicted_status"].map(urgency_rank)
    result = result.sort_values(["priority_rank", "confidence"], ascending=[True, False])
    return result.drop(columns=["priority_rank"])


@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_single(payload: dict):
    """One household, submitted as JSON."""
    raw_df = pd.DataFrame([payload])
    result = score_households(raw_df)
    row = result.iloc[0]
    return JSONResponse({
        "predicted_status": row["predicted_status"],
        "confidence": float(row["confidence"]),
    })


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """Many households at once, as a CSV upload. Returns a ranked CSV,
    most urgent households first -- this is the NGO targeting list."""
    raw_bytes = await file.read()
    raw_df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    result = score_households(raw_df)

    buf = io.StringIO()
    result.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prioritized_households.csv"},
    )
