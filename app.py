import streamlit as st

from forecast import (
    load_model,
    load_feature_columns,
    load_lag_history,
    forecast_n_months
)

from scenario import apply_dimension_scenario


st.set_page_config(
    page_title="Revenue Forecast Simulator",
    layout="wide"
)

st.title("Revenue Forecast Simulator")


@st.cache_resource
def load_assets():
    model = load_model("models/revenue_model.pkl")
    feature_cols = load_feature_columns("models/feature_columns.pkl")
    lag_history = load_lag_history("data/lag_history.parquet")
    lag_history = lag_history[lag_history["CUSTOMER_TYPE"] != "Unknown"].reset_index(drop=True)
    return model, feature_cols, lag_history


model, feature_cols, lag_history = load_assets()

st.subheader("Run Baseline Forecast")

n_months = st.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=12,
    value=6
)

if st.button("Run Revenue Forecast"):
    forecast_df = forecast_n_months(
        model=model,
        feature_columns=feature_cols,
        lag_history=lag_history,
        n_months=n_months
    )
    st.session_state["baseline_forecast"] = forecast_df

if "baseline_forecast" in st.session_state:

    forecast_df = st.session_state["baseline_forecast"]

    monthly_totals = (
        forecast_df.groupby("DATE")[["predicted_amount"]]
        .sum()
        .reset_index()
        .sort_values("DATE")
    )

    st.subheader("Baseline Forecast (Total Revenue)")
    st.line_chart(monthly_totals.set_index("DATE"))

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast (CSV)",
        data=csv,
        file_name="baseline_forecast.csv",
        mime="text/csv"
    )

    st.subheader("Scenario Explorer (Next Month Impact)")
    st.caption(
        "Explore how customer type or major category mix may impact next month’s forecast."
    )

    first_date = monthly_totals.iloc[0]["DATE"]
    month1_df = forecast_df[forecast_df["DATE"] == first_date].copy()

    active_dim = st.radio(
        "Adjust revenue segment",
        ["CUSTOMER_TYPE", "MAJOR_CATEGORY_BUCKET"],
        horizontal=True
    )

    categories = sorted(
        month1_df[active_dim].dropna().astype(str).unique()
    )

    pct_map = {}

    for cat in categories:
        pct_map[cat] = st.slider(
            f"{cat} (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=1,
            key=f"{active_dim}_{cat}"
        )

    if st.button("Apply Scenario"):

        if all(v == 0 for v in pct_map.values()):
            st.warning(
                "All sliders are 0%. Adjust at least one value to see scenario impact."
            )
        else:
            result = apply_dimension_scenario(
                forecast_df=month1_df,
                dimension=active_dim,
                pct_map=pct_map
            )

            st.markdown("### Scenario Result (Month 1)")

            col1, col2, col3 = st.columns(3)

            col1.metric("Baseline", f"{result['baseline_total']:,.0f}")
            col2.metric("Impact", f"{result['total_impact']:,.0f}")
            col3.metric("New Forecast", f"{result['new_total']:,.0f}")
