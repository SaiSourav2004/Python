import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Auto EDA Tool", layout="wide", page_icon="📊")
st.title("📊 Automated Data Visualization & EDA Tool")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(file):
    file_extension = file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            return pd.read_csv(file)
        elif file_extension == 'xlsx':
            return pd.read_excel(file)
        elif file_extension == 'json':
            return pd.read_json(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ------------------ SESSION ------------------
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

# ------------------ HELPERS ------------------
def summarize_data(df):
    return {
        "shape": df.shape,
        "missing": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum()
    }

def handle_missing(df, num_strategy, cat_strategy):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

        if df[col].isnull().sum() > 0:

            if pd.api.types.is_numeric_dtype(df[col]):
                if num_strategy == "Mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif num_strategy == "Median":
                    df[col] = df[col].fillna(df[col].median())
                elif num_strategy == "Drop":
                    df = df.dropna(subset=[col])
            else:
                if cat_strategy == "Mode":
                    if not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode()[0])
                elif cat_strategy == "Drop":
                    df = df.dropna(subset=[col])
    return df

def remove_outliers(df):
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

# ------------------ PLOTS ------------------
def plot_count(df, col, palette):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=col, data=df, palette=palette, ax=ax)
    plt.xticks(rotation=45)
    return fig

def plot_hist(df, col):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[col], kde=True, ax=ax)
    return fig

def plot_box(df, col):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x=df[col], ax=ax)
    return fig

def plot_scatter(df, x, y):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=x, y=y, data=df, ax=ax)
    return fig

def plot_heatmap(df, cmap):
    fig, ax = plt.subplots(figsize=(8,6))
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        ax.text(0.5, 0.5, "Not enough numeric columns", ha='center')
        return fig
    sns.heatmap(num_df.corr(), annot=True, cmap=cmap, ax=ax)
    return fig

# ------------------ FILE ------------------
file = st.file_uploader("Upload Dataset", type=["csv","xlsx","json"])

if file:
    df = load_data(file)

    if df is not None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        tab1, tab2, tab3 = st.tabs(["📁 Summary","🧹 Cleaning","📊 Visualization"])

        # -------- SUMMARY --------
        with tab1:
            st.dataframe(df.head())

            summary = summarize_data(df)
            c1,c2,c3 = st.columns(3)
            c1.metric("Shape", f"{summary['shape'][0]} x {summary['shape'][1]}")
            c2.metric("Missing", summary["missing"])
            c3.metric("Duplicates", summary["duplicates"])

            st.write("### Column Types")
            st.write(df.dtypes)

            st.write("### Statistical Summary")
            st.write(df.describe())

        # -------- CLEANING --------
        with tab2:
            num_strategy = st.selectbox("Numerical Strategy", ["Mean","Median","Drop"])
            cat_strategy = st.selectbox("Categorical Strategy", ["Mode","Drop"])

            remove_dup = st.checkbox("Remove Duplicates", True)
            remove_outliers_opt = st.checkbox("Remove Outliers", False)

            if st.button("Clean Data"):
                cleaned_df = df.copy()
                cleaned_df = handle_missing(cleaned_df, num_strategy, cat_strategy)

                if remove_dup:
                    cleaned_df = cleaned_df.drop_duplicates()

                if remove_outliers_opt:
                    cleaned_df = remove_outliers(cleaned_df)

                st.session_state.cleaned_df = cleaned_df
                st.success("Data Cleaned!")

            if st.session_state.cleaned_df is not None:
                st.dataframe(st.session_state.cleaned_df.head())

        # -------- VISUALIZATION --------
        with tab3:
            plot_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df

            # Categorical
            st.subheader("Categorical Analysis")
            if cat_cols:
                col = st.selectbox("Column", cat_cols)
                palette = st.selectbox("Theme", ["Set2","coolwarm","pastel"])
                st.pyplot(plot_count(plot_df, col, palette))

            # Numerical
            st.subheader("Numerical Analysis")
            if num_cols:
                col = st.selectbox("Numeric Column", num_cols)
                chart = st.selectbox("Chart Type", ["Histogram","Boxplot"])

                if chart=="Histogram":
                    st.pyplot(plot_hist(plot_df, col))
                else:
                    st.pyplot(plot_box(plot_df, col))

            # Bivariate
            st.subheader("Bivariate Analysis")
            if len(num_cols)>=2:
                x = st.selectbox("X", num_cols)
                y = st.selectbox("Y", num_cols, index=1)
                st.pyplot(plot_scatter(plot_df, x, y))

            # Heatmap
            st.subheader("Correlation Heatmap")
            cmap = st.selectbox("Theme", ["coolwarm","viridis","magma"])
            if st.button("Generate Heatmap"):
                st.pyplot(plot_heatmap(plot_df, cmap))
