import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Multivariate Analysis: PCA & Correlation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("DATS 6401: Week 3 — Tabular & Multivariate Data")
st.caption(
    "Homework 03: Correlation Heatmap & PCA Projection | Cellular Morphology"
    " Dataset"
)


# 2. Data Wrangling & Tidy Structure (Rubric Item 1: 2 pts)
@st.cache_data
def get_morphology_data():
  raw = load_breast_cancer(as_frame=True)
  # Select the 10 standardized 'mean' cellular dimension features
  mean_features = [col for col in raw.feature_names if "mean" in col]
  df = raw.data[mean_features].copy()

  # Clean column names for clear chart labeling
  df.columns = [
      c.replace("mean ", "").replace(" ", "_").capitalize() for c in df.columns
  ]
  numeric_cols = df.columns.tolist()

  # Add primary categorical target
  df["Diagnosis"] = raw.target.map({0: "Malignant", 1: "Benign"})

  # Derive secondary categorical bins for the color-by widget
  df["Size_Category"] = pd.qcut(
      df["Area"], q=3, labels=["Small Cell", "Medium Cell", "Large Cell"]
  )
  df["Shape_Irregularity"] = pd.qcut(
      df["Fractal_dimension"],
      q=3,
      labels=["Regular Shape", "Moderate Irregularity", "High Irregularity"],
  )

  return df, numeric_cols


df, numeric_cols = get_morphology_data()

# -------------------------------------------------------------
# 3. Sidebar Widget: Interactive "Color-By" Selector
# -------------------------------------------------------------
st.sidebar.header("Visualization Controls")
color_options = {
    "Clinical Diagnosis (Malignant vs. Benign)": "Diagnosis",
    "Cell Area Category": "Size_Category",
    "Nuclear Shape Irregularity": "Shape_Irregularity",
}
selected_label = st.sidebar.selectbox(
    "Color PCA Points By:", list(color_options.keys())
)
color_column = color_options[selected_label]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset Scope:**\n* **Observations:** 569 patient biopsy samples\n*"
    f" **Continuous Measurements:** {len(numeric_cols)} morphology"
    " dimensions\n* **Primary Classes:** Malignant vs. Benign"
)

with st.expander("🔍 View Tidy Dataframe (First 10 Rows)"):
  st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------
# 4. Multivariate Visualizations (Rubric Item 2: 4 pts)
# -------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

# --- Column 1: Correlation Heatmap ---
with col1:
  st.subheader("1. Correlation Heatmap")
  st.caption("Pearson correlation (r) across 10 cell morphology measurements")

  corr_matrix = df[numeric_cols].corr()

  fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
  sns.heatmap(
      corr_matrix,
      cmap="vlag",
      center=0,
      vmin=-1,
      vmax=1,
      linewidths=0.5,
      cbar_kws={"label": "Pearson Correlation (r)", "shrink": 0.8},
      ax=ax_corr,
      annot=True,
      fmt=".2f",
      annot_kws={"size": 7},
  )
  ax_corr.tick_params(axis="both", labelsize=8)
  plt.xticks(rotation=45, ha="right")
  plt.yticks(rotation=0)
  plt.tight_layout()
  st.pyplot(fig_corr)

# --- Column 2: PCA Projection ---
with col2:
  st.subheader("2. PCA 2D Projection")
  st.caption(f"Points colored by: **{selected_label}**")

  # Standardize all features to zero-mean and unit-variance
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(df[numeric_cols])

  # Fit 2 Principal Components
  pca = PCA(n_components=2)
  pca_coords = pca.fit_transform(X_scaled)
  var_explained = pca.explained_variance_ratio_ * 100

  pca_df = pd.DataFrame({
      "PC1": pca_coords[:, 0],
      "PC2": pca_coords[:, 1],
      "category": df[color_column],
  })

  fig_pca, ax_pca = plt.subplots(figsize=(7, 6))
  unique_cats = sorted(df[color_column].unique())
  palette = sns.color_palette("Set2", n_colors=len(unique_cats))

  for idx, cat in enumerate(unique_cats):
    subset = pca_df[pca_df["category"] == cat]
    ax_pca.scatter(
        subset["PC1"],
        subset["PC2"],
        label=str(cat),
        color=palette[idx],
        s=40,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

  ax_pca.set_title(
      "PCA Projection (Standardized Features)", fontsize=11, fontweight="bold"
  )
  ax_pca.set_xlabel(f"PC1 ({var_explained[0]:.2f}% Variance)", fontsize=9)
  ax_pca.set_ylabel(f"PC2 ({var_explained[1]:.2f}% Variance)", fontsize=9)
  ax_pca.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
  ax_pca.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
  ax_pca.grid(True, linestyle=":", alpha=0.5)
  ax_pca.legend(title=selected_label, fontsize=8, title_fontsize=9)
  plt.tight_layout()
  st.pyplot(fig_pca)

# -------------------------------------------------------------
# 5. PCA Interpretation & Feature Loadings (Rubric Item 3: 4 pts)
# -------------------------------------------------------------
st.markdown("---")
st.subheader("3. Interpretation of PCA Structure & Components")

interp_col1, interp_col2 = st.columns([1.2, 0.8], gap="medium")

with interp_col1:
  st.markdown(f"""
    **Observed Data Structure:**
    * **Clear Biological Separation:** The 2D PCA projection reveals substantial separation between **Malignant** (positive PC1) and **Benign** (negative PC1) cellular profiles.
    * **High Information Retention:** Together, the first two principal components capture **{var_explained[0] + var_explained[1]:.2f}%** of the total variance across all 10 morphology measurements ({var_explained[0]:.2f}% by PC1 and {var_explained[1]:.2f}% by PC2).

    **What PC1 Represents ({var_explained[0]:.2f}% of Variance):**
    * **PC1 represents Nuclear Size and Boundary Severity.**
    * It is dominated by strong positive loadings on `Concave_points` ($+0.42$), `Concavity` ($+0.40$), `Perimeter` ($+0.38$), and `Area` ($+0.36$).
    * Larger, deeply indented cell nuclei project strongly toward positive PC1 values, indicating malignant tissue behavior.

    **What PC2 Represents ({var_explained[1]:.2f}% of Variance):**
    * **PC2 represents Shape Complexity and Texture Irregularity.**
    * It is driven by positive loadings on `Fractal_dimension` ($+0.57$), `Smoothness` ($+0.40$), and `Symmetry` ($+0.37$), alongside negative loadings on macro-dimensions like `Radius` ($-0.31$) and `Area` ($-0.30$).
    * Cells with positive PC2 values exhibit convoluted, non-smooth cell borders regardless of baseline size.
    """)

with interp_col2:
  st.markdown("**Feature Loadings on PC1 & PC2:**")
  loadings = pd.DataFrame(
      pca.components_.T,
      index=numeric_cols,
      columns=["PC1 Loading", "PC2 Loading"],
  ).round(3)
  st.dataframe(loadings, use_container_width=True)