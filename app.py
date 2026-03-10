import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
        pca_cols = st.multiselect("Select columns for PCA", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])

        # Select column for coloring
        color_col = st.selectbox("Select column for coloring points", df.columns)

        if pca_cols:
            # Handle missing values by dropping rows with NaN in selected columns
            df_pca = df.dropna(subset=pca_cols)
            if df_pca.empty:
                st.error("No valid data after removing missing values.")
            else:
                X = df_pca[pca_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=2)
                pcs = pca.fit_transform(X_scaled)

                pc_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=df_pca.index)
                pc_df[color_col] = df_pca[color_col]

                # Variance explained
                var_explained = pca.explained_variance_ratio_ * 100
                st.write(f"**Variance Explained:** PC1: {var_explained[0]:.2f}%, PC2: {var_explained[1]:.2f}%")

                # Plot controls
                title = st.text_input("Plot Title", "PCA Plot")
                xlabel = st.text_input("X-axis Label", "PC1")
                ylabel = st.text_input("Y-axis Label", "PC2")

                # Generate plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue=color_col, ax=ax, palette='viridis' if pc_df[color_col].dtype == 'object' else None)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                st.pyplot(fig)
        else:
            st.warning("Please select at least one column for PCA.")
else:
    st.info("Please upload a CSV file to get started.")
