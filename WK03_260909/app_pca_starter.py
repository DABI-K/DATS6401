# app_pca_starter.py — Week 3: PCA explorer with a color-by selector
# Run with:  streamlit run app_pca_starter.py
import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("PCA Explorer")

@st.cache_data
def load_and_project():
    wine = load_wine(as_frame=True)
    X = StandardScaler().fit_transform(wine.data)
    pca = PCA(2)
    pcs = pca.fit_transform(X)
    df = wine.data.copy()
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
    df["cultivar"] = wine.target.astype(str)
    return df, pca.explained_variance_ratio_

df, evr = load_and_project()
st.caption(f"PC1 {evr[0]:.0%} · PC2 {evr[1]:.0%} of variance")

# TODO: add a selectbox over the columns and use it as the color:
# color_col = st.selectbox("Color points by", ["cultivar"] + list(df.columns[:13]))
color_col = "cultivar"

st.scatter_chart(df, x="PC1", y="PC2", color=color_col)
