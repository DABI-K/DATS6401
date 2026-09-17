# app_rolling_animation.py — Week 4: watch a rolling window do its work
# Run with:  streamlit run app_rolling_animation.py
#
# The point of this app is to make `series.rolling(w).mean()` concrete. The grey
# band is the window. It slides left to right, and at each step the mean of the
# values inside it is added to the smooth line below.
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Rolling window", layout="wide")
st.title("What a rolling window actually does")


@st.cache_data
def load_monthly():
    co2 = sm.datasets.co2.load_pandas().data.dropna()
    return co2["co2"].resample("MS").mean().dropna().loc["1980":"2000"]


s = load_monthly()


def draw(s, rolled, window, upto: int):
    """One frame: the series, the current window, and the smooth line so far."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(s.index, s.values, color="#c9d3da", lw=1.0, label="monthly")

    if upto >= window:
        lo, hi = s.index[upto - window], s.index[upto - 1]
        ax.axvspan(lo, hi, color="#2E6E8E", alpha=0.18)
        ax.plot(rolled.index[:upto], rolled.values[:upto],
                color="#2E6E8E", lw=2.2, label=f"rolling({window}).mean()")
        ax.plot([rolled.index[upto - 1]], [rolled.values[upto - 1]],
                "o", color="#d9534f", ms=7)

    ax.set_ylim(s.min() - 2, s.max() + 2)
    ax.set_ylabel("CO₂ (ppm)")
    ax.legend(loc="upper left")
    ax.set_title(f"window = {window} months")
    fig.tight_layout()
    return fig


# Everything interactive lives in a fragment. Without it, a button click reruns
# the whole script, and Streamlit leaves the previous run's chart on screen,
# greyed out, for the entire length of the animation — so you watch a duplicate.
@st.fragment
def rolling_panel(s):
    c1, c2, c3 = st.columns([2, 2, 1])
    window = c1.slider("Window length (months)", 2, 60, 12,
                       help="12 is one full seasonal cycle for monthly data.")
    n_frames = c2.slider("Frames", 20, 120, 60,
                         help="More frames is smoother but takes longer to play.")
    play = c3.button("▶ Play", use_container_width=True)

    rolled = s.rolling(window).mean()
    frame = st.empty()

    if play:
        # Each frame re-renders the figure and ships a PNG to the browser, so
        # redraw cost sets the pace — there is no sleep here, and adding one only
        # makes an already slow loop slower.
        step = max(1, (len(s) - window) // n_frames)
        for upto in range(window, len(s) + 1, step):
            fig = draw(s, rolled, window, upto)
            frame.pyplot(fig)
            plt.close(fig)      # without this, figures accumulate and Streamlit warns

    fig = draw(s, rolled, window, len(s))
    frame.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"The smooth line starts {window - 1} months in: a window of {window} has "
        "no value until it is full, which is why `rolling` returns NaN at the start."
    )


rolling_panel(s)

with st.expander("Things to try"):
    st.markdown(
        """
- Set the window to **3** and play. The seasonal sawtooth survives, because the
  window is shorter than the 12-month cycle.
- Set it to **12** and play. Each window covers exactly one cycle, so the
  seasonality averages out and only the trend is left.
- Set it to **60**. The line is smoother still, but genuine turning points are
  flattened along with the noise.
- Watch the left edge as you change the window: the larger the window, the later
  the smooth line can start.
        """
    )
