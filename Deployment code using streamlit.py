#!/usr/bin/env python
# coding: utf-8

# In[28]:


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, SpectralClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# --- Page Setup ---
st.set_page_config(page_title="Clustering Dashboard", layout="wide")
st.sidebar.title("🔧 Controls")

# --- File Upload or Sample Dataset ---
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ File '{uploaded_file.name}' loaded successfully.")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    st.info("No file uploaded. Using sample dataset.")
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target.astype(str)

# --- Data Cleaning ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)
df.drop_duplicates(inplace=True)

# --- EDA ---
st.title("📊 Exploratory Data Analysis")
st.dataframe(df.head())
st.dataframe(df.describe())

# Missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    st.write("Missing Values", missing[missing > 0])
else:
    st.write("No missing values.")

# Correlation heatmap (sampled if large)
if len(numeric_cols) > 1:
    sample_corr = df[numeric_cols].sample(min(500, len(df))) if len(df) > 500 else df
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(sample_corr.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Column distribution
col = st.selectbox("Select column for distribution", numeric_cols)
fig, ax = plt.subplots()
sns.histplot(df[col], kde=True, ax=ax)
st.pyplot(fig)

# --- Clustering Setup ---
st.header("🤖 Clustering Model Evaluation")
selected_features = st.sidebar.multiselect(
    "Select features for clustering", numeric_cols, default=numeric_cols.tolist()
)

if not selected_features:
    st.warning("Please select at least one numeric feature.")
    st.stop()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[selected_features])

sample_size = st.sidebar.slider("Sample size for model testing", min_value=50, max_value=len(scaled_data), value=min(100, len(scaled_data)))
sample_data = scaled_data[:sample_size]

# Algorithm parameters
dbscan_eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5, 0.1)
dbscan_min = st.sidebar.slider("DBSCAN min_samples", 3, 20, 5)
optics_min = st.sidebar.slider("OPTICS min_samples", 3, 20, 5)

# Select algorithms
algo_options = st.sidebar.multiselect(
    "Select clustering algorithms to run",
    ["KMeans", "Agglomerative", "GaussianMixture", "Birch", "SpectralClustering", "OPTICS", "DBSCAN"],
    default=["KMeans", "DBSCAN"]
)

# Models dict
models = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=3),
    "GaussianMixture": GaussianMixture(n_components=3, random_state=42),
    "Birch": Birch(n_clusters=3),
    "SpectralClustering": SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=42),
    "OPTICS": OPTICS(min_samples=optics_min),
    "DBSCAN": DBSCAN(eps=dbscan_eps, min_samples=dbscan_min)
}

best_model = None
best_score = -1
best_labels = None
model_scores = {}
skipped_models = {}

with st.spinner("🔍 Evaluating clustering models..."):
    for name in algo_options:
        model = models[name]
        try:
            labels = model.fit_predict(sample_data)
            valid_labels = [l for l in set(labels) if l != -1]
            if len(valid_labels) > 1:
                score = silhouette_score(sample_data, labels)
                model_scores[name] = score
                if score > best_score:
                    best_score = score
                    best_model = name
                    best_labels = model.fit_predict(scaled_data)
            else:
                model_scores[name] = None
                skipped_models[name] = "Too few clusters or noise"
        except Exception as e:
            model_scores[name] = None
            skipped_models[name] = str(e)

# Results
st.subheader("📈 Silhouette Scores")
st.dataframe(pd.DataFrame.from_dict(model_scores, orient="index", columns=["Silhouette Score"]))

if skipped_models:
    st.warning("⚠️ Skipped Models")
    for k, v in skipped_models.items():
        st.markdown(f"- {k}: {v}")

if best_model is None:
    st.error("❌ No valid clustering model found.")
    st.stop()

st.success(f"✅ Best Model Selected: **{best_model}** with Silhouette Score: **{round(best_score, 3)}**")
df["Cluster"] = best_labels

# --- Cluster Visualizations ---
st.header("📍 Cluster Visualizations")
pca = PCA(n_components=2).fit_transform(scaled_data)
fig = px.scatter(x=pca[:,0], y=pca[:,1], color=df["Cluster"].astype(str), title="2D PCA Cluster Plot")
st.plotly_chart(fig)

if "Country" in df.columns:
    fig = px.scatter_geo(df, locations="Country", locationmode="country names",
                         color="Cluster", title="Country-wise Cluster Distribution")
    st.plotly_chart(fig)

# Cluster summary
st.subheader("📋 Cluster Summary")
st.write(df.groupby("Cluster")[selected_features].mean())

# Cluster profiles
st.subheader("🧠 Cluster Profiles")
for cluster_id in sorted(df["Cluster"].unique()):
    st.markdown(f"### Cluster {cluster_id}")
    cluster_data = df[df["Cluster"] == cluster_id]
    top_features = cluster_data[selected_features].mean().sort_values(ascending=False).head(3)
    st.write("Top features:", top_features.index.tolist())
    st.write(cluster_data.describe().T)

# --- Export ---
filename = f"clustered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
st.download_button("📥 Download Clustered Data", df.to_csv(index=False).encode("utf-8"), filename, "text/csv")


# In[ ]:




