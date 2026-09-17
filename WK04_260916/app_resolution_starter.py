# # app_resolution_starter.py — Week 4: the resolution lesson as a widget
# # Run with:  streamlit run app_resolution_starter.py
# import streamlit as st
# import pandas as pd
# import statsmodels.api as sm

# st.title("Resolution Changes the Story")

# @st.cache_data
# def load_series():
#     co2 = sm.datasets.co2.load_pandas().data.dropna()
#     return co2["co2"]

# series = load_series()

# # TODO: add a selectbox over resample rules ["W", "MS", "QS", "YS"]
# # and resample `series` with the chosen rule before plotting:
# rule = "MS"                              # <- replace with the selectbox
# shown = series.resample(rule).mean()

# window = st.slider("Rolling window (periods)", 1, 36, 12)

# st.line_chart(pd.DataFrame({
#     "raw": shown,
#     f"rolling({window})": shown.rolling(window).mean(),
# }))
# st.caption(f"Resolution: {rule} · window: {window} — both choices belong in your caption.")







import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import streamlit as st

st.set_page_config(
    page_title="Resolution Changes the Story",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Resolution Changes the Story")
st.caption(
    "Week 4: Time Series Dynamics, Resolution Trade-offs, and Uncertainty"
)


@st.cache_data
def load_series():
  # Load raw CO2 and interpolate across sporadic missing dates
  raw = sm.datasets.co2.load_pandas().data
  s = raw["co2"].interpolate(method="time")
  s.index = pd.to_datetime(s.index)
  return s


series = load_series()

# -------------------------------------------------------------
# 1. Widgets: Resolution Selector & Rolling Window
# -------------------------------------------------------------
rule_labels = {
    "W": "Weekly (High Resolution)",
    "MS": "Month Start (Balanced Cycle)",
    "QS": "Quarter Start (Smoothed)",
    "YS": "Year Start (Macro Trend Only)",
}

rule = st.selectbox(
    "Select Resampling Resolution:",
    options=["W", "MS", "QS", "YS"],
    index=1,
    format_func=lambda x: f"{x} — {rule_labels[x]}",
)

window = st.slider(
    "Rolling window (periods based on selected resolution)", 1, 36, 12
)

# Resample according to user selection
shown = series.resample(rule).mean().dropna()

# -------------------------------------------------------------
# 2. Main Visualizations: Two Columns
# -------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
  st.subheader("1. Interactive Trend & Rolling Mean")
  rolling_series = shown.rolling(window, center=True).mean()

  chart_df = pd.DataFrame(
      {
          "Resampled Series": shown,
          f"Rolling Mean ({window} periods)": rolling_series,
      }
  )
  st.line_chart(chart_df)
  st.caption(
      f"Resolution: **{rule}** ({rule_labels[rule]}) · Rolling Window:"
      f" **{window} periods**."
  )

with col2:
  st.subheader("2. Uncertainty Interval (±2σ Empirical Variance)")
  st.caption("Centered rolling mean with empirical confidence spread")

  # Calculate rolling standard deviation for uncertainty estimation
  rolling_std = shown.rolling(window, center=True).std()
  upper_band = rolling_series + (2 * rolling_std)
  lower_band = rolling_series - (2 * rolling_std)

  fig_unc, ax_unc = plt.subplots(figsize=(7, 4.2))
  ax_unc.plot(
      shown.index,
      shown,
      color="lightgray",
      linewidth=0.8,
      label=f"Resampled ({rule})",
  )
  ax_unc.plot(
      rolling_series.index,
      rolling_series,
      color="#d62728",
      linewidth=1.8,
      label=f"Trend ({window}p)",
  )
  ax_unc.fill_between(
      rolling_series.index,
      lower_band,
      upper_band,
      color="#d62728",
      alpha=0.2,
      label="Uncertainty (±2σ)",
  )

  ax_unc.set_ylabel("CO2 (ppm)")
  ax_unc.grid(True, linestyle="--", alpha=0.5)
  ax_unc.legend(loc="upper left", fontsize=8)
  plt.tight_layout()
  st.pyplot(fig_unc)

# -------------------------------------------------------------
# 3. Additive Seasonality Decomposition
# -------------------------------------------------------------
st.markdown("---")
st.subheader(
    "3. Classical Seasonal Decomposition (Monthly Baseline, Period = 12)"
)

monthly_series = series.resample("MS").mean().dropna()
decomp = seasonal_decompose(monthly_series, model="additive", period=12)

fig_decomp, axes = plt.subplots(4, 1, figsize=(14, 7), sharex=True)
axes[0].plot(decomp.observed.index, decomp.observed, color="#1f77b4")
axes[0].set_ylabel("Observed")
axes[0].grid(True, linestyle="--", alpha=0.4)

axes[1].plot(decomp.trend.index, decomp.trend, color="#d62728")
axes[1].set_ylabel("Trend")
axes[1].grid(True, linestyle="--", alpha=0.4)

axes[2].plot(decomp.seasonal.index, decomp.seasonal, color="#2ca02c")
axes[2].set_ylabel("Seasonal")
axes[2].grid(True, linestyle="--", alpha=0.4)

axes[3].plot(
    decomp.resid.index, decomp.resid, color="#7f7f7f", linewidth=0.7
)
axes[3].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[3].set_ylabel("Residual")
axes[3].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
st.pyplot(fig_decomp)

# -------------------------------------------------------------
# 4. Temporal-Honesty Disclosure Note
# -------------------------------------------------------------
st.markdown("---")
st.subheader("Temporal-Honesty Design Choices")
st.info("""
* **Non-Zero Vertical Baseline:** The $y$-axis starts near $310\\text{ ppm}$ rather than $0\\text{ ppm}$. In atmospheric monitoring, zero is outside the meaningful physical domain. Starting at zero would flatten the multi-decade rise and annual oscillations into an unreadable horizontal line. The truncated axis is explicitly disclosed through clearly labeled tick marks.
* **Centered Rolling Windows:** The moving average uses `center=True` to prevent the 6-month phase lag that occurs in trailing moving averages, aligning peaks and valleys with their true calendar timing.
* **Resolution Disclosures:** Resampling to annual intervals (`YS`) removes the seasonal photosynthetic cycle entirely, while weekly resolution (`W`) introduces short-term weather noise. Switching between them demonstrates how temporal aggregation choices can either conceal or emphasize cyclical phenomena.
""")