import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def compute_pca(df, pca_cols):
    df_pca = df.dropna(subset=pca_cols).drop_duplicates()
    if df_pca.empty:
        return None
    X = df_pca[pca_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    pc_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=df_pca.index)
    var_explained = pca.explained_variance_ratio_ * 100
    return pc_df, var_explained, df_pca

st.title("PCA Visualization App")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Select numeric columns for PCA
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in the data.")
    else:
        use_all = st.checkbox("Use all numeric columns for PCA", value=True)
        if use_all:
            pca_cols = numeric_cols
        else:
            pca_cols = st.multiselect("Select columns for PCA", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])

        # Select column for coloring
        color_col = st.selectbox("Select column for colour coding", df.columns)

        # Plot controls
        title = st.text_input("Plot Title", "PCA Plot")
        xlabel = st.text_input("X-axis Label", "PC1")
        ylabel = st.text_input("Y-axis Label", "PC2")
        palette = st.selectbox("Color Palette", ["viridis", "plasma", "coolwarm", "Set1", "tab10"], index=0)
        opacity = st.slider("Point Opacity", 0.1, 1.0, 0.8)

        if st.button("Generate PCA Plot"):
            if pca_cols:
                result = compute_pca(df, pca_cols)
                if result is None:
                    st.error("No valid data after removing missing values and duplicates.")
                else:
                    pc_df, var_explained, df_pca = result
                    pc_df[color_col] = df_pca[color_col]

                    # Variance explained
                    st.write(f"**Variance Explained:** PC1: {var_explained[0]:.2f}%, PC2: {var_explained[1]:.2f}%")

                    # Generate plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue=color_col, ax=ax, palette=palette, alpha=opacity)
                    ax.set_title(title)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    st.pyplot(fig)
            else:
                st.warning("Please select at least one column for PCA.")
else:
    st.info("Please upload a CSV file to get started.")
