# conda activate dataviz
# streamlit run hw02_app.py

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# -------------------------------------------------------------
# Configuration & Theming
# -------------------------------------------------------------
st.set_page_config(
    page_title="EV Sales: Visual Encodings & Perceptual Accuracy",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("DATS 6401: Grammar of Graphics & Design Principles")
st.caption(
    "Eungdap Kim | HW02 | Fall 2026"
)

# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------
data = {
    "Year": [2020, 2021, 2022, 2023, 2024],
    "Tesla": [500, 936, 1314, 1808, 1810],
    "BYD": [180, 593, 1863, 3024, 3400],
    "Volkswagen Group": [220, 453, 572, 771, 740],
    "BMW Group": [193, 328, 434, 566, 620],
    "Geely": [68, 100, 329, 672, 850],
}
df = pd.DataFrame(data)

df_long = df.melt(
    id_vars=["Year"], var_name="Manufacturer", value_name="Sales_Thousands"
)

# -------------------------------------------------------------
# Context / Assignment Header
# -------------------------------------------------------------
st.markdown(
    """
> **Underlying Question:** *How did annual global sales volume and market trajectories evolve among the top five EV manufacturers from 2020 to 2024?*
"""
)

# -------------------------------------------------------------
# Three Encodings Side-by-Side
# -------------------------------------------------------------
col1, col2, col3 = st.columns(3, gap="medium")

# =============================================================
# ENCODING 1: Multi-Line Chart (Position on a Common Scale)
# =============================================================
with col1:
    st.subheader("Encoding 1: Common-Scale Position")
    st.caption("Channels: X-Position (Year) · Y-Position (Sales) · Hue (Brand)")

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    palette = sns.color_palette("tab10", n_colors=len(df.columns[1:]))

    for idx, maker in enumerate(df.columns[1:]):
        ax1.plot(
            df["Year"],
            df[maker],
            marker="o",
            markersize=5,
            linewidth=2,
            label=maker,
            color=palette[idx],
        )

    ax1.set_title(
        "Trajectory by Brand (Aligned Scale)", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("Year", fontsize=9)
    ax1.set_ylabel("Sales (Thousands of Units)", fontsize=9)
    ax1.set_xticks(df["Year"])
    ax1.tick_params(axis="both", labelsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(title="Brand", fontsize=7, title_fontsize=8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown(
        """
    **Perceptual Accuracy Reasoning:**
    * Encodes quantitative sales via **position along a common aligned scale** ($y$-axis).
    * According to Cleveland & McGill's ranking, position on a common scale enables the most accurate quantitative comparisons. Observers judge both exact values and rates of change (slopes) with minimal cognitive bias.

    **Gestalt / Preattentive Principle:**
    * **Gestalt Principle of Connectedness & Continuity:** Explicit line marks physically connect discrete annual points into a coherent visual flow, prompting the viewer to perceive each brand as an unbroken trajectory over time.
    """
    )

# =============================================================
# ENCODING 2: Stacked Bar Chart (Length on Unaligned Scales)
# =============================================================
with col2:
    st.subheader("Encoding 2: Unaligned Length")
    st.caption(
        "Channels: X-Position (Year) · Segment Length (Sales) · Hue (Brand)"
    )

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bottom_tracker = np.zeros(len(df["Year"]))

    for idx, maker in enumerate(df.columns[1:]):
        values = df[maker].values
        ax2.bar(
            df["Year"],
            values,
            bottom=bottom_tracker,
            label=maker,
            width=0.55,
            edgecolor="white",
            color=palette[idx],
        )
        bottom_tracker += values

    ax2.set_title(
        "Total & Segment Share (Unaligned)", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("Year", fontsize=9)
    ax2.set_ylabel("Sales (Thousands of Units)", fontsize=9)
    ax2.set_xticks(df["Year"])
    ax2.tick_params(axis="both", labelsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(title="Brand", fontsize=7, title_fontsize=8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown(
        """
    **Perceptual Accuracy Reasoning:**
    * Encodes individual automaker volumes using **length along unaligned scales** (except the baseline brand).
    * Cleveland & McGill demonstrated that length without a common baseline induces significant perceptual degradation; readers must estimate segment boundaries without a shared origin, making year-over-year comparisons for inner segments (e.g., Volkswagen, BMW) substantially less accurate.

    **Gestalt / Preattentive Principle:**
    * **Gestalt Principle of Proximity:** Vertical stacking forces parts into a unified column group, emphasizing annual market totals over individual brand evolutions.
    """
    )

# =============================================================
# ENCODING 3: Matrix Heatmap (Color Saturation / Luminance)
# =============================================================
with col3:
    st.subheader("Encoding 3: Color Saturation")
    st.caption(
        "Channels: Matrix Position (Brand × Year) · Color Luminance (Sales)"
    )

    heatmap_matrix = df.set_index("Year").T

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar_kws={"label": "Sales (k)"},
        ax=ax3,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax3.set_title(
        "Sales Intensity Matrix (Color Scale)", fontsize=11, fontweight="bold"
    )
    ax3.set_xlabel("Year", fontsize=9)
    ax3.set_ylabel("Brand", fontsize=9)
    ax3.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown(
        """
    **Perceptual Accuracy Reasoning:**
    * Encodes numerical values using **color saturation and luminance**.
    * Color saturation sits near the bottom of Cleveland & McGill’s hierarchy for quantitative magnitude. The human eye cannot reliably perform ratio estimation or derive precise differences between subtle variations in shading without referencing numerical text annotations.

    **Gestalt / Preattentive Principle:**
    * **Preattentive Visual Attribute (Intensity):** Deep navy tones (BYD 2023–2024) trigger preattentive pop-out within 200 ms, guiding ocular focus to volume peaks before active reading begins.
    """
    )

# -------------------------------------------------------------
# Rubric Reflection & Summary
# -------------------------------------------------------------
st.markdown("---")
st.subheader("Perceptual Accuracy Ranking (Cleveland & McGill Synthesis)")
comparison_table = pd.DataFrame(
    {
        "Encoding": [
            "Chart 1: Line Plot",
            "Chart 2: Stacked Bar",
            "Chart 3: Heatmap",
        ],
        "Primary Quantitative Channel": [
            "Position along a common scale",
            "Length on unaligned scales",
            "Color saturation / luminance",
        ],
        "Cleveland–McGill Hierarchy Position": [
            "Tier 1 (Highest Accuracy)",
            "Tier 3 (Moderate Accuracy)",
            "Tier 6 (Lowest Accuracy)",
        ],
        "Primary Cognitive Strength": [
            "Instant slope/trend & ratio reading",
            "Macro-level aggregate total volume",
            "Preattentive detection of peak hotspots",
        ],
    }
)
st.dataframe(comparison_table, use_container_width=True, hide_index=True)