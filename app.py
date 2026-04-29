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

# ------------------ SESSION STATE ------------------
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
        # Convert possible numeric strings to numbers
        df[col] = pd.to_numeric(df[col], errors='ignore')

        if df[col].isnull().sum() > 0:

            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):

                if num_strategy == "Mean":
                    df[col] = df[col].fillna(df[col].mean())

                elif num_strategy == "Median":
                    df[col] = df[col].fillna(df[col].median())

                elif num_strategy == "Drop":
                    df = df.dropna(subset=[col])

            # Categorical columns
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

# ------------------ PLOTTING ------------------
def plot_count(df, col):
    fig, ax = plt.subplots()
    sns.countplot(x=col, data=df, ax=ax)
    plt.xticks(rotation=45)
    return fig

def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
    return fig

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("Upload Dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if file:
    df = load_data(file)

    if df is not None:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        tab1, tab2, tab3 = st.tabs(["📁 Summary", "🧹 Cleaning", "📊 Visualization"])

        # -------- SUMMARY --------
        with tab1:
            st.write("### Data Preview")
            st.dataframe(df.head())

            summary = summarize_data(df)
            st.write("### Dataset Info")
            st.write(f"Shape: {summary['shape']}")
            st.write(f"Missing Values: {summary['missing']}")
            st.write(f"Duplicates: {summary['duplicates']}")

            st.write("### Column Types")
            st.write(df.dtypes)

            st.write("### Statistical Summary")
            st.write(df.describe())

        # -------- CLEANING --------
        with tab2:
            st.subheader("Data Cleaning")

            num_strategy = st.selectbox("Numerical Strategy", ["Mean", "Median", "Drop"])
            cat_strategy = st.selectbox("Categorical Strategy", ["Mode", "Drop"])

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
                st.success("Data cleaned successfully!")

            if st.session_state.cleaned_df is not None:
                st.dataframe(st.session_state.cleaned_df.head())

                csv = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Clean Data", csv, "cleaned.csv")

        # -------- VISUALIZATION --------
        with tab3:
            plot_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df

            st.subheader("Categorical Analysis")
            if cat_cols:
                cat_col = st.selectbox("Select column", cat_cols)
                if cat_col:
                    st.pyplot(plot_count(plot_df, cat_col))

            st.subheader("Correlation Heatmap")
            if st.button("Show Heatmap"):
                st.pyplot(plot_heatmap(plot_df))
