import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Auto EDA Tool", layout="wide", page_icon="📊")
st.title("📊 Automated Data Visualization & EDA Tool")

# ------------------ CACHED FUNCTIONS ------------------
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
        st.error(f"Error loading file: {e}. Please check the file format.")
        return None
    
@st.cache_data
def generate_profile(df):
    profile = ProfileReport(df, explorative=True, minimal=True)
    return profile.to_html()

# ------------------ SESSION STATE ------------------
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

# ------------------ HELPER FUNCTIONS ------------------
def summarize_data(df):
    return {
        "shape": df.shape,
        "missing": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum()
    }

def handle_missing(df, num_strategy, cat_strategy):
    cols_to_drop = []
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                if cat_strategy == "Mode" and not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif cat_strategy == "Drop":
                    cols_to_drop.append(col)
            else:
                if num_strategy == "Median":
                    df[col] = df[col].fillna(df[col].median())
                elif num_strategy == "Mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif num_strategy == "Drop":
                    cols_to_drop.append(col)
                    
    # Safely drops columns with missing values at the end
    if cols_to_drop:
        df = df.dropna(subset=cols_to_drop)
        
    return df

def remove_outliers(df):
    # Automatically applies IQR outlier removal to all numerical columns
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[
            (df[col] >= Q1 - 1.5 * IQR) & 
            (df[col] <= Q3 + 1.5 * IQR)
        ]
    return df

# ------------------ PLOTTING FUNCTIONS ------------------
def plot_count(df, col, theme):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=col, data=df, ax=ax, hue=col, palette=theme, legend=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    return fig

def plot_pie(df, col, theme):
    fig, ax = plt.subplots(figsize=(6, 6))
    counts = df[col].value_counts()
    
    # Groups small categories into 'Other' if there are more than 10 to keep pie chart clean
    if len(counts) > 10:
        top_counts = counts.iloc[:9]
        top_counts['Other'] = counts.iloc[9:].sum()
        counts = top_counts
        
    colors = sns.color_palette(theme, len(counts)) 
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal') 
    return fig

def plot_heatmap(df, cmap_choice):
    num_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(8, 6))
    if num_df.shape[1] < 2:
        ax.text(0.5, 0.5, "Not enough numeric columns", ha='center', va='center')
        return fig
    sns.heatmap(num_df.corr(), annot=True, cmap=cmap_choice, ax=ax, fmt=".2f", linewidths=.5)
    fig.tight_layout()
    return fig

def plot_bivariate(df, col1, col2, plot_type):
    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_type == "Scatter Plot":
        sns.scatterplot(x=col1, y=col2, data=df, ax=ax, alpha=0.7)
    elif plot_type == "Line Plot":
        sns.lineplot(x=col1, y=col2, data=df, ax=ax)
    fig.tight_layout()
    return fig

def interpret_count(df, col):
    top = df[col].value_counts().idxmax()
    return f"Most frequent category in '{col}' is '{top}'."

def interpret_heatmap(corr):
    strong_pairs = []
    for col in corr.columns:
        for idx in corr.index:
            if col != idx and abs(corr.loc[idx, col]) > 0.7:
                strong_pairs.append((idx, col, corr.loc[idx, col]))
    if not strong_pairs:
        return "No strong correlations found (Threshold > 0.7)."
    
    unique_pairs = set([tuple(sorted([a, b]) + [val]) for a, b, val in strong_pairs])
    insights = "**Strong correlations:**\n"
    for a, b, val in unique_pairs:
        insights += f"* {a} vs {b}: {val:.2f}\n"
    return insights

# ------------------ MAIN APP ------------------
# Updated file uploader to accept multiple formats
file = st.file_uploader("Upload Dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if file:
    df = load_data(file)
    
    if df is not None:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Summary", "🧹 Data Cleaning", "📈 Visualizations", "📄 Full Report"])

        # --- TAB 1: SUMMARY ---
        with tab1:
            st.write("### Raw Data Preview")
            st.dataframe(df.head())

            summary = summarize_data(df)
            st.write("### Data Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows & Columns", f"{summary['shape'][0]} x {summary['shape'][1]}")
            col2.metric("Total Duplicates", summary["duplicates"])
            col3.metric("Total Missing Values", summary["missing"])

        # --- TAB 2: CLEANING ---
        with tab2:
            st.subheader("🧹 Data Cleaning Options")
            
            c1, c2 = st.columns(2)
            with c1:
                num_strategy = st.selectbox("Numerical Imputation Strategy", ["Median", "Mean", "Drop"])
            with c2:
                cat_strategy = st.selectbox("Categorical Imputation Strategy", ["Mode", "Drop"])

            remove_dup = st.checkbox("Remove Duplicates", True)
            remove_outliers_opt = st.checkbox("Remove Outliers (IQR Method)", False)

            if st.button("Clean Data", type="primary"):
                cleaned_df = df.copy()
                cleaned_df = handle_missing(cleaned_df, num_strategy, cat_strategy)
                
                if remove_outliers_opt:
                    cleaned_df = remove_outliers(cleaned_df)
                
                if remove_dup:
                    cleaned_df = cleaned_df.drop_duplicates()

                st.session_state.cleaned_df = cleaned_df
                st.success("Data cleaned successfully!")

            if st.session_state.cleaned_df is not None:
                st.write("### Cleaned Data Preview")
                st.dataframe(st.session_state.cleaned_df.head())
                st.write("New Shape:", st.session_state.cleaned_df.shape)

                csv = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Cleaned Dataset", data=csv, file_name="cleaned_data.csv", mime="text/csv")

        # --- TAB 3: VISUALIZATIONS ---
        with tab3:
            plot_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df
            dataset_status = "Cleaned" if st.session_state.cleaned_df is not None else "Raw"
            st.info(f"Visualizations are currently using the **{dataset_status}** dataset.")

            try:
                # 1. Categorical Analysis 
                with st.expander("📊 Categorical Analysis", expanded=True):
                    selected_cat = st.selectbox("Select a categorical column", ["None"] + cat_cols)
                    if selected_cat != "None":
                        cat_ui_cols = st.columns([2, 1])
                        with cat_ui_cols[0]:
                            cat_plot_type = st.radio("Select Plot Type", ["Bar Chart", "Pie Chart"], horizontal=True)
                        with cat_ui_cols[1]:
                            cat_theme = st.selectbox("Color Theme", ["Set2", "Pastel1", "husl", "muted", "Paired", "Dark2"])
                        
                        if cat_plot_type == "Bar Chart":
                            fig = plot_count(plot_df, selected_cat, cat_theme)
                        else:
                            fig = plot_pie(plot_df, selected_cat, cat_theme)
                            
                        st.pyplot(fig)
                        st.write(interpret_count(plot_df, selected_cat))

                # 2. Numerical Analysis
                with st.expander("📈 Numerical Analysis"):
                    selected_num = st.multiselect("Select numerical columns", num_cols)
                    num_plot_type = st.selectbox("Select plot type", ["Histogram", "Boxplot"])

                    if selected_num:
                        for col in selected_num:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            if num_plot_type == "Histogram":
                                sns.histplot(plot_df[col], kde=True, ax=ax, color='skyblue')
                            elif num_plot_type == "Boxplot":
                                sns.boxplot(x=plot_df[col], ax=ax, color='lightgreen')
                            
                            st.write(f"**{num_plot_type} of {col}**")
                            st.pyplot(fig)

                # 3. Bivariate Analysis
                with st.expander("🔍 Bivariate Analysis (X vs Y)"):
                    if len(num_cols) >= 2:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_axis = st.selectbox("X-axis", num_cols, key="biv_x")
                        with col2:
                            y_axis = st.selectbox("Y-axis", num_cols, index=1, key="biv_y")
                        with col3:
                            biv_plot_type = st.selectbox("Plot Type", ["Scatter Plot", "Line Plot"])

                        fig = plot_bivariate(plot_df, x_axis, y_axis, biv_plot_type)
                        st.pyplot(fig)
                        st.write(f"This plot shows the relationship between **{x_axis}** and **{y_axis}**.")
                    else:
                        st.warning("Not enough numerical columns for bivariate analysis.")

                # 4. Correlation Heatmap
                with st.expander("🔥 Correlation Heatmap"):
                    heatmap_cols = st.columns([1, 3])
                    with heatmap_cols[0]:
                        cmap_choice = st.selectbox("Heatmap Theme", ["coolwarm", "viridis", "magma", "crest", "Blues", "Reds", "YlGnBu"])
                    
                    if st.checkbox("Generate Correlation Heatmap"):
                        fig = plot_heatmap(plot_df, cmap_choice)
                        st.pyplot(fig)
                        corr = plot_df.select_dtypes(include='number').corr()
                        st.write(interpret_heatmap(corr))
                        
            except Exception as e:
                st.error(f"An error occurred while generating visualizations: {e}. Please check your data structure.")

        # --- TAB 4: EDA REPORT ---
        with tab4:
            st.write("Generate a comprehensive automated HTML report using `ydata-profiling`.")
            report_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df
            
            if st.button("Generate EDA Report", type="primary"):
                with st.spinner("Generating complex report... This may take a minute."):
                    # Limits row count to prevent the presentation computer from crashing
                    if len(report_df) > 10000:
                        st.warning("Dataset is large. Profiling a random sample of 10,000 rows to ensure stability.")
                        report_df = report_df.sample(10000, random_state=42)
                        
                    report_html = generate_profile(report_df)
                    st.components.v1.html(report_html, height=800, scrolling=True)
                    st.download_button("📥 Download Full HTML Report", data=report_html, file_name="Automated_EDA_Report.html", mime="text/html")