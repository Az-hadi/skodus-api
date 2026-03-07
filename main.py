import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import io

app = FastAPI(title="Skodus Engine", version="1.0.0")






MAX_FILE_SIZE = 10 * 1024 * 1024

@app.get("/health")
def health():
    return {"status": "ok", "brand": "SKODUS", "engine": "v1.0.0"}

@app.post("/analyze")
async def skodus_engine(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Maximum 10MB.")
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Invalid CSV file.")
    if len(df.columns) < 2:
        raise HTTPException(400, "CSV needs at least 2 columns.")
    if len(df) < 10:
        raise HTTPException(400, "Not enough rows (minimum 10).")

    df = df.ffill().bfill()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise HTTPException(400, "No numeric column found.")

    target = numeric_cols[-1]
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    results = {"brand": "SKODUS", "status": "verified", "target_column": target}

    if date_cols:
        results["analysis_type"] = "Sales Forecasting"
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df['m'] = df[date_cols[0]].dt.month
        df['y'] = df[date_cols[0]].dt.year
        X, y = df[['m', 'y']], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        last_date = df[date_cols[0]].max()
        future_dates = pd.date_range(last_date, periods=4, freq='MS')[1:]
        preds = model.predict(pd.DataFrame([[d.month, d.year] for d in future_dates], columns=['m', 'y']))
        results["metrics"] = {"confidence_score": max(0, round(model.score(X_test, y_test), 2))}
        results["forecast"] = [round(p, 2) for p in preds.tolist()]
        results["forecast_periods"] = [str(d.date()) for d in future_dates]
        results["summary"] = "Growth trajectory computed on seasonal cycles."
    else:
        results["analysis_type"] = "Financial Risk Analysis"
        df_enc = df.copy()
        le = LabelEncoder()
        for col in df_enc.select_dtypes(include=['object']).columns:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        X, y = df_enc.drop(columns=[target]), df_enc[target]
        if y.nunique() < 2:
            raise HTTPException(400, "Target needs at least 2 distinct values.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        drivers = sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:3]
        results["metrics"] = {"accuracy": round(model.score(X_test, y_test), 2)}
        results["risk_drivers"] = {name: round(val, 3) for name, val in drivers}
        results["summary"] = "Risk score established. Key anomaly vectors identified."

    return results
