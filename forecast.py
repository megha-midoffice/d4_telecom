# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 20:08:55 2026

@author: MeghaGhosh_b5xx485
"""

import pickle
import pandas as pd
import numpy as np


FEATURE_COLS = [
    "lag_1",
    "lag_2",
    "roll3_mean",
    "mom_change",
    "roll3_std"
]

CAT_COLS = [
    "CUSTOMER_TYPE",
    "MAJOR_CATEGORY_BUCKET"
]

SEG_KEYS = CAT_COLS


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_columns(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_lag_history(path):
    return pd.read_parquet(path)


def forecast_n_months(
    model,
    feature_columns,
    lag_history,
    n_months=1
):
    lag_history = lag_history.copy()
    lag_history["DATE"] = pd.to_datetime(lag_history["DATE"])
    lag_history = lag_history.sort_values(SEG_KEYS + ["DATE"])

    last_date = lag_history["DATE"].max()

    segment_history = {}

    for seg, g in lag_history.groupby(SEG_KEYS):
        vals = g.sort_values("DATE")["AMOUNT"].tail(3).tolist()
        vals = list(reversed(vals))
        segment_history[seg] = vals

    all_forecasts = []
    current_date = last_date

    for _ in range(n_months):

        forecast_date = (current_date + pd.DateOffset(months=1)).replace(day=1)

        rows = []

        for seg, history_vals in segment_history.items():

            lag_1 = history_vals[0] if len(history_vals) > 0 else 0
            lag_2 = history_vals[1] if len(history_vals) > 1 else 0

            vals = history_vals[:3]
            while len(vals) < 3:
                vals.append(0)

            roll3_mean = np.mean(vals)
            roll3_std = np.std(vals)
            mom_change = lag_1 - lag_2

            row = dict(zip(SEG_KEYS, seg))
            row.update({
                "DATE": forecast_date,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "roll3_mean": roll3_mean,
                "roll3_std": roll3_std,
                "mom_change": mom_change
            })

            rows.append(row)

        df_features = pd.DataFrame(rows)

        X = pd.get_dummies(df_features[FEATURE_COLS + CAT_COLS], drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)

        preds = model.predict(X)
        df_features["predicted_amount"] = preds

        all_forecasts.append(
            df_features[
                SEG_KEYS + [
                    "DATE",
                    "predicted_amount"
                ]
            ]
        )

        for i, seg in enumerate(segment_history.keys()):
            pred_val = preds[i]
            old_vals = segment_history[seg]
            segment_history[seg] = [pred_val] + old_vals[:2]

        current_date = forecast_date

    return pd.concat(all_forecasts, ignore_index=True)
