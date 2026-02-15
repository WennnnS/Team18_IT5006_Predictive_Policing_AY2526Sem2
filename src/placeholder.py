import pandas as pd
import numpy as np

def fake_df():
    # 一个最小可用的假数据，用于先把dashboard跑通
    rng = pd.date_range("2024-01-01", periods=500, freq="H")
    df = pd.DataFrame({
        "Date": rng,
        "Primary Type": np.random.choice(["THEFT", "BATTERY", "BURGLARY"], size=len(rng)),
        "Latitude": 41.88 + np.random.normal(0, 0.03, size=len(rng)),
        "Longitude": -87.63 + np.random.normal(0, 0.03, size=len(rng)),
    })
    df["year"] = df["Date"].dt.year
    df["hour"] = df["Date"].dt.hour
    return df
