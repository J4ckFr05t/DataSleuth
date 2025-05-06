import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from rapidfuzz import process
import re
import streamlit.components.v1 as components
import io
import base64
from collections import Counter

st.set_page_config(page_title="🧠 Smart EDA Viewer", layout="wide", initial_sidebar_state="expanded")

# Dark mode style
dark_style = """
<style>
body {
    background-color: #121212;
    color: #e0e0e0;
}
.stButton>button {
    background-color: #333 !important;
    color: white !important;
}
</style>
"""
st.markdown(dark_style, unsafe_allow_html=True)

st.title("📊 Smart EDA Viewer")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Sidebar inputs
st.sidebar.header("🌍 Country & Region Config")
countries_input = st.sidebar.text_area("Country List (comma separated)", value="USA, India, Germany, France, UK, Canada")
regions_input = st.sidebar.text_area("Region List (comma separated)", value="California, Bavaria, Ontario, Tamil Nadu")

COUNTRY_LIST = [x.strip() for x in countries_input.split(",")]
REGION_LIST = [x.strip() for x in regions_input.split(",")]


def simplify_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype): return "int"
    if pd.api.types.is_float_dtype(dtype): return "float"
    if pd.api.types.is_string_dtype(dtype): return "string"
    if pd.api.types.is_bool_dtype(dtype): return "bool"
    if pd.api.types.is_datetime64_any_dtype(dtype): return "datetime"
    return "other"


def extract_country_region(text, country_list, region_list):
    text = str(text).lower()
    country_match = process.extractOne(text, country_list, score_cutoff=80)
    region_match = process.extractOne(text, region_list, score_cutoff=80)
    return {
        "country": country_match[0] if country_match else None,
        "region": region_match[0] if region_match else None
    }


def detect_pattern(value):
    known_patterns = {
        r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$": "IPv4",
        r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$": "MAC Address",
        r"^\S+@\S+\.\S+$": "Email",
        r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z]{2,})+$": "FQDN",
    }
    for pat, label in known_patterns.items():
        if re.match(pat, str(value)): return label

    # Abstract pattern: A=uppercase, a=lowercase, 9=digit, @=special
    pattern = ""
    for char in str(value):
        if char.isupper(): pattern += "A"
        elif char.islower(): pattern += "a"
        elif char.isdigit(): pattern += "9"
        elif re.match(r"\W", char): pattern += "@"
        else: pattern += "?"
    return pattern


if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.success(f"✅ Loaded **{df.shape[0]}** records with **{df.shape[1]}** fields.")

    st.subheader("🧾 Field-wise Summary")
    summaries = []
    for col in df.columns:
        dtype = simplify_dtype(df[col].dtype)
        total = len(df)
        nulls = df[col].isnull().sum()
        null_pct = round((nulls / total) * 100, 2)
        distinct = df[col].nunique()
        coverage = 100 - null_pct
        summaries.append({"Field": col, "Type": dtype, "Coverage %": coverage, "Nulls": nulls, "Distinct Values": distinct})

    summary_df = pd.DataFrame(summaries)
    st.dataframe(summary_df, use_container_width=True)

    with st.expander("📥 Export Summary"):
        csv = summary_df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="eda_summary.csv">📄 Download as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        html = summary_df.to_html(index=False)
        st.download_button("📄 Download as Confluence-compatible HTML", data=html, file_name="eda_summary.html", mime="text/html")

    st.subheader("📌 Per Field Insights")
    for col in df.columns:
        st.markdown(f"### 🧬 {col}")
        col_data = df[col].dropna()
        total = len(df)
        coverage = 100 - (df[col].isnull().sum() / total * 100)
        nunique = col_data.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        if nunique == total:
            with st.expander("📋 View Unique Values (with counts)"):
                value_counts = col_data.value_counts().reset_index()
                value_counts.columns = ["Value", "Count"]
                value_counts = value_counts.sort_values("Count", ascending=False)
                st.dataframe(value_counts, use_container_width=True)
        elif not is_numeric:
            val_counts = col_data.value_counts()
            if nunique <= 50:
                fig, ax = plt.subplots(figsize=(8, min(0.3 * len(val_counts), 10)))
                sns.barplot(x=val_counts.values, y=val_counts.index, ax=ax, palette="viridis")
                ax.set_title("Top Values")
                st.pyplot(fig)
            else:
                st.caption(f"📊 Showing top 10 of {nunique} unique values")
                top10 = val_counts.head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=top10.values, y=top10.index, ax=ax, palette="magma")
                ax.set_title("Top 10 Values")
                st.pyplot(fig)
                with st.expander("📋 Full Value Counts"):
                    st.dataframe(val_counts.reset_index().rename(columns={"index": "Value", col: "Count"}), use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(col_data, kde=True, color="teal", ax=ax)
            ax.set_title("Distribution")
            st.pyplot(fig)

        st.progress(int(coverage), text=f"Coverage: {coverage:.2f}%")

    st.subheader("🗝️ Potential Primary Keys")
    single_keys = [col for col in df.columns if df[col].is_unique and df[col].notnull().all()]
    st.write("Single column keys:", single_keys if single_keys else "None")
    composite_keys = []
    for i in range(2, 4):
        for combo in combinations(df.columns, i):
            if df[list(combo)].dropna().drop_duplicates().shape[0] == df.shape[0]:
                composite_keys.append(combo)
    st.write("Composite keys:", composite_keys if composite_keys else "None")

    st.subheader("🔗 Correlation Analysis (Numerical Fields Only)")
    st.caption("This section shows how strongly numerical fields are related to each other. Values close to +1/-1 mean strong correlation.")
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] > 1:
        corr = num_df.corr().round(2)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title("Correlation Matrix", fontsize=14)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric fields for correlation analysis.")

    st.subheader("🌍 Country/Region Detection (Text Columns)")
    extraction_summary = {}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        results = df[col].dropna().apply(lambda x: extract_country_region(x, COUNTRY_LIST, REGION_LIST))
        country_hits = results.apply(lambda x: x['country']).dropna().unique()
        region_hits = results.apply(lambda x: x['region']).dropna().unique()
        extraction_summary[col] = {
            "countries_found": list(country_hits),
            "regions_found": list(region_hits)
        }
    for col, hits in extraction_summary.items():
        st.markdown(f"**{col}**")
        st.write(f"• Countries: {', '.join(hits['countries_found']) if hits['countries_found'] else 'None'}")
        st.write(f"• Regions: {', '.join(hits['regions_found']) if hits['regions_found'] else 'None'}")

    st.subheader("🔍 Pattern Detection")
    st.markdown("""
    Each value is scanned for **known formats** like IP, MAC, Email, FQDN. 
    Other values are classified into **abstract patterns** using:
    - **A** = Uppercase letter
    - **a** = Lowercase letter
    - **9** = Digit
    - **@** = Special character
    - **?** = Other
    """)
    pattern_summary = []
    for col in df.columns:
        col_data = df[col].dropna().astype(str)
        patterns = col_data.apply(detect_pattern)
        top_patterns = patterns.value_counts(normalize=True).head(5)
        dominant_pct = top_patterns.iloc[0] * 100 if not top_patterns.empty else 0
        pattern_summary.append({
            "Field": col,
            "Top Pattern": top_patterns.index[0] if not top_patterns.empty else None,
            "% Values Match": round(dominant_pct, 2),
            "Flag (<80%)": "⚠️" if dominant_pct < 80 else "✅"
        })

    pattern_df = pd.DataFrame(pattern_summary)
    st.dataframe(pattern_df, use_container_width=True)

    with st.expander("📤 Export Pattern Detection Results"):
        csv = pattern_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", data=csv, file_name="patterns.csv", mime="text/csv")
        html = pattern_df.to_html(index=False)
        st.download_button("📄 Download HTML", data=html, file_name="patterns.html", mime="text/html")
else:
    st.info("📂 Please upload a file to begin analysis.")