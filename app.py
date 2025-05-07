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
import numpy as np

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
countries_input = st.sidebar.text_area("Country List (comma separated)", value="India, Bharat, Republic of India, United Arab Emirates, UAE, Emirates, Saudi Arabia, KSA, Kingdom of Saudi Arabia, United Kingdom, UK, Britain, Great Britain, United States of America, USA, US, United States, America, Armenia, Republic of Armenia, Azerbaijan, Republic of Azerbaijan, Canada, Côte d'Ivoire, Ivory Coast, Chile, Republic of Chile, Colombia, Republic of Colombia, Costa Rica, Republic of Costa Rica, Germany, Deutschland, Federal Republic of Germany, Ecuador, Republic of Ecuador, Egypt, Arab Republic of Egypt, Spain, España, Kingdom of Spain, France, French Republic, Georgia, Sakartvelo, Ghana, Republic of Ghana, Croatia, Republic of Croatia, Italy, Italian Republic, Japan, Nippon, Nihon, Republic of Korea, South Korea, Korea (South), Lithuania, Republic of Lithuania, Luxembourg, Grand Duchy of Luxembourg, Morocco, Kingdom of Morocco, TFYR Macedonia, North Macedonia, Macedonia, Mexico, United Mexican States, Netherlands, Holland, Kingdom of the Netherlands, Philippines, Republic of the Philippines, Peru, Republic of Peru, Poland, Republic of Poland, Portugal, Portuguese Republic, Romania, Senegal, Republic of Senegal, Suriname, Republic of Suriname, Togo, Togolese Republic, Thailand, Kingdom of Thailand, Siam, Turkey, Türkiye, Republic of Turkey, Ethiopia, Federal Democratic Republic of Ethiopia, Algeria, People’s Democratic Republic of Algeria, Jordan, Hashemite Kingdom of Jordan, Madagascar, Republic of Madagascar, Kazakhstan, Republic of Kazakhstan, China, People’s Republic of China, PRC, Lebanon, Lebanese Republic, Serbia, Republic of Serbia, South Africa, Republic of South Africa, United Republic of Tanzania, Tanzania, Cameroon, Republic of Cameroon, Russian Federation, Russia, Switzerland, Swiss Confederation, Viet Nam, Vietnam, Socialist Republic of Vietnam, Nigeria, Federal Republic of Nigeria, Indonesia, Republic of Indonesia, Uganda, Republic of Uganda, Ukraine, Rwanda, Republic of Rwanda, Gabon, Gabonese Republic, Belarus, Kenya, Republic of Kenya, Kosovo, Republic of Kosovo, Tunisia, Republic of Tunisia, Uzbekistan, Republic of Uzbekistan, Albania, Republic of Albania, Jamaica, CTSS, Argentina, Argentine Republic, Australia, Commonwealth of Australia, Bosnia and Herzegovina, BiH, Belgium, Kingdom of Belgium, Brazil, Federative Republic of Brazil, Czech Republic, Czechia, Denmark, Kingdom of Denmark, Dominican Republic, Finland, Republic of Finland, Greece, Hellenic Republic, Mauritius, Republic of Mauritius, Guatemala, Republic of Guatemala, Guyana, Co-operative Republic of Guyana, Honduras, Republic of Honduras, Ireland, Éire, Republic of Ireland, Malaysia, Nicaragua, Republic of Nicaragua, Norway, Kingdom of Norway, Sweden, Kingdom of Sweden, Singapore, Republic of Singapore, El Salvador, Republic of El Salvador, Estonia, Republic of Estonia")
regions_input = st.sidebar.text_area("Region List (comma separated)", value="APAC, EMEA, EWAP, Global, INDIA, LATAM, MAJOREL, Specialized Services, TGI")

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

    # Direct substring match
    matched_countries = [c for c in country_list if c.lower() in text]
    matched_regions = [r for r in region_list if r.lower() in text]

    # Fallback to fuzzy match if no substring hit
    if not matched_countries:
        fuzzy_country = process.extractOne(text, country_list, score_cutoff=80)
        if fuzzy_country:
            matched_countries.append(fuzzy_country[0])

    if not matched_regions:
        fuzzy_region = process.extractOne(text, region_list, score_cutoff=80)
        if fuzzy_region:
            matched_regions.append(fuzzy_region[0])

    return {
        "countries": matched_countries,
        "regions": matched_regions
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
    num_df = df.select_dtypes(include="number")

    if num_df.shape[1] > 1:
        corr_matrix = num_df.corr()
        abs_corr = corr_matrix.abs()

        upper = abs_corr.where(~np.tril(np.ones(abs_corr.shape)).astype(bool))
        top_pairs = (
            upper.stack()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: "Correlation"})
        )

        st.markdown("### 🔝 Top 10 Strongest Correlations")
        st.dataframe(top_pairs.head(10), use_container_width=True)

        perfect_corr = top_pairs[top_pairs["Correlation"] == 1.0]
        if not perfect_corr.empty:
            st.warning(f"⚠️ Perfect correlations detected: {len(perfect_corr)} pairs. This suggests redundant features or multicollinearity.")

        strong_corr_cols = abs_corr.columns[(abs_corr > 0.6).any()]
        if len(strong_corr_cols) >= 2:
            st.markdown("### 🔥 Heatmap of Strong Correlations (|corr| > 0.6)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr_matrix.loc[strong_corr_cols, strong_corr_cols],
                annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": .5}, ax=ax
            )
            ax.set_title("Strong Correlation Heatmap", fontsize=14)
            st.pyplot(fig)
        else:
            st.info("No strong correlations (|corr| > 0.6) found between numerical fields.")
    else:
        st.info("Not enough numeric fields for correlation analysis.")

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

    st.subheader("🌍 Country/Region Extraction Insights")
    extraction_summary = {}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        results = df[col].dropna().apply(lambda x: extract_country_region(x, COUNTRY_LIST, REGION_LIST))
        all_countries = [c for res in results for c in res['countries']]
        all_regions = [r for res in results for r in res['regions']]
        country_hits = list(set(all_countries))
        region_hits = list(set(all_regions))

        extraction_summary[col] = {
            "countries_found": country_hits,
            "regions_found": region_hits
        }

    for col, hits in extraction_summary.items():
        st.markdown(f"**{col}**")
        st.write(f"• Countries: {', '.join(hits['countries_found']) if hits['countries_found'] else 'None'}")
        st.write(f"• Regions: {', '.join(hits['regions_found']) if hits['regions_found'] else 'None'}")

else:
    st.info("📂 Please upload a file to begin analysis.")
