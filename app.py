import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import re
import streamlit.components.v1 as components
import io
import base64
from collections import Counter
import numpy as np
import re
import ahocorasick
import matplotlib.ticker as mticker


st.set_page_config(page_title="DataSleuth", layout="wide", initial_sidebar_state="expanded")

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

st.title("📊 DataSleuth - Smart EDA Viewer")

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

def detect_pattern(value):
    value = str(value).strip()
    known_patterns = {
        r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$": ("IPv4", "<octet>.<octet>.<octet>.<octet>"),
        r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$": ("MAC Address", "<hex>:<hex>:<hex>:<hex>:<hex>:<hex>"),
        r"^([a-zA-Z0-9_.+-]+)\@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)$": ("Email", "<username>@<domain>.<tld>"),
        r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z]{2,})+$": ("FQDN", "<subdomain>.<domain>.<tld>"),
    }
    for pat, (label, example) in known_patterns.items():
        if re.match(pat, value):
            return label, example

    # Symbolic abstraction: A=uppercase, a=lowercase, 9=digit, @=special
    symbol_map = {
        'A': lambda c: c.isupper(),
        'a': lambda c: c.islower(),
        '9': lambda c: c.isdigit(),
        '@': lambda c: re.match(r"\W", c) and not c.isspace()
    }
    pattern = ""
    for char in value:
        for symbol, cond in symbol_map.items():
            if cond(char):
                pattern += symbol
                break
        else:
            pattern += '?'
    return pattern, None

AMBIGUOUS_TERMS = {"us", "uk", "ctss", "eire","apache"}

@st.cache_resource
def build_automaton(keyword_list):
    A = ahocorasick.Automaton()
    for idx, word in enumerate(keyword_list):
        A.add_word(word.lower(), (idx, word))
    A.make_automaton()
    return A

# Build automatons once
COUNTRY_AUTOMATON = build_automaton(COUNTRY_LIST)
REGION_AUTOMATON = build_automaton(REGION_LIST)

def is_valid_match(term, text):
    term_l = term.lower()
    if term_l in AMBIGUOUS_TERMS:
        return re.search(rf'\\b{re.escape(term_l)}\\b', text.lower()) is not None
    return True

def extract_country_region(text, *_):
    text_lower = str(text).lower()
    countries = set()
    regions = set()

    for _, (_, match) in COUNTRY_AUTOMATON.iter(text_lower):
        if is_valid_match(match, text):
            countries.add(match)

    for _, (_, match) in REGION_AUTOMATON.iter(text_lower):
        if is_valid_match(match, text):
            regions.add(match)

    return {
        "countries": list(countries),
        "regions": list(regions)
    }

def shorten_labels(labels, max_len=50): 
    """Shortens labels to a maximum length, reserving space for '...'."""
    return [label if len(label) <= max_len else label[:max_len - 3] + '...' for label in labels]

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

    st.subheader("🗝️ Primary Key Identification")

    # Step 1: Auto-detect single-column primary keys
    auto_keys = [col for col in df.columns if df[col].is_unique and df[col].notnull().all()]
    has_auto_keys = bool(auto_keys)

    if has_auto_keys:
        st.write("🔍 Automatically Detected Primary Key(s):", auto_keys)
        user_choice = st.radio(
            "Are you happy with the automatically detected primary key(s)?",
            ["Yes", "No", "Skip"],
            horizontal=True
        )
    else:
        st.warning("❗ No single-column primary keys were automatically detected.")
        user_choice = st.radio(
            "No primary key was detected. Do you want to manually choose a key or skip?",
            ["Choose manually", "Skip"],
            horizontal=True
        )

    # Step 2: Handle response
    if user_choice == "Yes":
        st.success(f"✅ Using auto-detected primary key(s): {auto_keys}")
    elif user_choice in ["No", "Choose manually"]:
        st.info("🔧 Manually choose one or more columns to form a primary key.")
        manual_keys = st.multiselect(
            "Select one or more fields (case-insensitive)",
            options=df.columns.tolist()
        )
        if manual_keys:
            if df[manual_keys].dropna().drop_duplicates().shape[0] == df.shape[0]:
                st.success(f"✅ Selected fields form a valid primary key: {manual_keys}")
            else:
                st.error("❌ Selected fields do not form a unique primary key.")
        else:
            st.warning("⚠️ No primary key selected. Proceeding without one.")
    else:
        st.info("➡️ Proceeding without setting a primary key.")

    total_records = df.shape[0]
    primary_keys = manual_keys if 'manual_keys' in locals() and manual_keys else auto_keys if 'auto_keys' in locals() and auto_keys else []
    st.markdown("### 📌 Dataset Summary")

    if primary_keys:
        unique_keys = df[primary_keys].dropna().drop_duplicates().shape[0]
        st.info(f"🔢 Total Records: **{total_records}**  🔑 Unique Primary Keys: **{unique_keys}** (based on `{', '.join(primary_keys)}`)")
    else:
        st.info(f"🔢 Total Records: **{total_records}**  🔑 Primary key not selected.")


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
            top_n = 10

            st.markdown("#### Chart A: Top Values (All Rows)")
            val_counts_total = df[col].value_counts().head(top_n)
            percent_total = (val_counts_total / total * 100).round(2)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=val_counts_total.values, y=shorten_labels(val_counts_total.index.tolist()), ax=ax, palette="Blues_d")
            ax.set_title("Top 10 Values (All Records)")
            ax.set_xlabel("Occurrences")
            ax.set_ylabel(col)
            st.pyplot(fig)

            # Create a table showing values, counts, and percentages
            percent_table = pd.DataFrame({
                "Value": val_counts_total.index,
                "Count": val_counts_total.values,
                "Percentage (%)": percent_total
            })

            # Add total count to the table
            percent_table = pd.concat([percent_table, pd.DataFrame([["Total", total, "100"]], columns=percent_table.columns)], ignore_index=True)

            st.markdown("### 📊 Value Counts and Percentages")
            st.dataframe(percent_table, use_container_width=True)

            # Determine primary key(s)
            primary_keys = manual_keys if 'manual_keys' in locals() and manual_keys else auto_keys if 'auto_keys' in locals() and auto_keys else []

            if primary_keys:
                try:
                    temp_df = df[primary_keys + [col]].dropna().drop_duplicates(subset=primary_keys)
                    grouped_counts = temp_df[col].value_counts().head(top_n)
                    percent_keys = (grouped_counts / temp_df.shape[0] * 100).round(2)

                    st.markdown("#### Chart B: Top Values (Per Primary Key)")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=grouped_counts.values, y=shorten_labels(grouped_counts.index.tolist()), ax=ax, palette="Greens_d")
                    ax.set_title("Top 10 Values (Per Unique Primary Key)")
                    ax.set_xlabel("Occurrences")
                    ax.set_ylabel(col)
                    st.pyplot(fig)

                    # Create a table showing per primary key values, counts, and percentages
                    percent_key_table = pd.DataFrame({
                        "Value": grouped_counts.index,
                        "Count": grouped_counts.values,
                        "Percentage (%)": percent_keys
                    })

                    # Add total count to the table
                    percent_key_table = pd.concat([percent_key_table, pd.DataFrame([["Total", temp_df.shape[0], "100"]], columns=percent_key_table.columns)], ignore_index=True)

                    st.markdown("### 📊 Value Counts and Percentages (Per Primary Key)")
                    st.dataframe(percent_key_table, use_container_width=True)

                except Exception as e:
                    st.error(f"⚠️ Error generating primary key-based chart: {e}")
            else:
                st.info("ℹ️ No primary key selected or detected, so Chart B is skipped.")

        else:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(col_data, kde=True, color="teal", ax=ax)
            ax.set_title("Distribution")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plt.tight_layout()
            st.pyplot(fig)

        st.progress(int(coverage), text=f"Coverage: {coverage:.2f}%")

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
    If not matched, a symbolic abstraction is used:
    - **A** = Uppercase letter
    - **a** = Lowercase letter
    - **9** = Digit
    - **@** = Special character
    - **?** = Other
    """)

    all_pattern_info = []

    for col in df.columns:
        col_data = df[col].dropna().astype(str)
        patterns = col_data.apply(detect_pattern)

        # Count pattern frequencies
        pattern_counts = patterns.apply(lambda x: x[0]).value_counts()
        total = pattern_counts.sum()

        pattern_info = []
        for pat, count in pattern_counts.items():
            example = patterns[patterns.apply(lambda x: x[0]) == pat].iloc[0][1]
            confidence = round((count / total) * 100, 2)
            pattern_info.append({
                "Field": col,
                "Pattern": pat,
                "Example": example if example else "",
                "Confidence (%)": confidence
            })

        all_pattern_info.extend(pattern_info)

    pattern_df = pd.DataFrame(all_pattern_info)

    st.markdown("### 📋 Detailed Pattern Report")
    st.dataframe(pattern_df, use_container_width=True)

    st.markdown("### 🌟 Most Common Pattern Per Field")
    top_patterns = pattern_df.sort_values('Confidence (%)', ascending=False).drop_duplicates('Field')
    st.dataframe(top_patterns[['Field', 'Pattern', 'Example', 'Confidence (%)']], use_container_width=True)

    with st.expander("📤 Export Pattern Detection Results"):
        csv = pattern_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", data=csv, file_name="all_patterns.csv", mime="text/csv")

        html = pattern_df.to_html(index=False)
        st.download_button("📄 Download HTML", data=html, file_name="all_patterns.html", mime="text/html")

    # Assuming extract_country_region and shorten_labels functions are defined elsewhere.

    st.subheader("🌍 Country/Region Extraction Insights")
    extraction_summary = {}

    # Collecting all country data for the summary table
    country_data = []

    # Get the total number of records in the DataFrame
    total_records = len(df)

    # Loop over each column that contains object or string types
    for col in df.select_dtypes(include=['object', 'string']).columns:
        # Get the non-null values
        non_null_values = df[col].dropna()
        
        # Use all the non-null records
        sampled_values = non_null_values  # No sampling, use all non-null records
        
        # Apply the extraction function to each non-null value
        results = sampled_values.apply(lambda x: (x, extract_country_region(x)))

        country_samples = {}

        for val, res in results:
            for c in res['countries']:
                if c not in country_samples:
                    country_samples[c] = val  # first occurrence

        # Store the country data in a summary format
        for country, sample in country_samples.items():
            count = results[results.apply(lambda x: country in x[1]['countries'])].shape[0]
            percentage = (count / total_records) * 100  # Calculate the percentage

            country_data.append({
                'Field': col,
                'Country': country,
                'Count': count,
                'Percentage': f"{percentage:.2f}%",  # Show percentage with 2 decimal places
                'Sample': sample,
                'Records Processed': len(sampled_values)  # Number of records sampled from the column
            })

    # Create a DataFrame to display the country extraction summary
    country_df = pd.DataFrame(country_data)

    # Display the consolidated summary table
    if not country_df.empty:
        st.write("### Country Extraction Summary")
        st.dataframe(country_df)
    else:
        st.write("No countries were extracted from the data.")
else:
    st.info("📂 Please upload a file to begin analysis.")
