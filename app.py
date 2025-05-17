import streamlit as st
import pandas as pd
import altair as alt
from itertools import combinations
import re
import base64
import ahocorasick
import pickle
from io import BytesIO
import os
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from lxml import etree
import xmltodict


def simplify_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype): return "int"
    if pd.api.types.is_float_dtype(dtype): return "float"
    if pd.api.types.is_string_dtype(dtype): return "string"
    if pd.api.types.is_bool_dtype(dtype): return "bool"
    if pd.api.types.is_datetime64_any_dtype(dtype): return "datetime"
    return "other"

def is_internal_ip(ip):
    """Check if an IP address is internal/private"""
    # Convert to list of integers
    try:
        octets = [int(x) for x in ip.split('.')]
        if len(octets) != 4:
            return False
        
        # Check for private IP ranges
        if octets[0] == 10:  # 10.0.0.0/8
            return True
        if octets[0] == 172 and 16 <= octets[1] <= 31:  # 172.16.0.0/12
            return True
        if octets[0] == 192 and octets[1] == 168:  # 192.168.0.0/16
            return True
        if octets[0] == 127:  # 127.0.0.0/8 (localhost)
            return True
        if octets[0] == 169 and octets[1] == 254:  # 169.254.0.0/16 (link-local)
            return True
        
        return False
    except:
        return False

def detect_pattern(value):
    value = str(value).strip()
    known_patterns = {
        r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$": ("IP Address", "<octet>.<octet>.<octet>.<octet>"),
        r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$": ("MAC Address", "<hex>:<hex>:<hex>:<hex>:<hex>:<hex>"),
        r"^([a-zA-Z0-9_.+-]+)\@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)$": ("Email", "<username>@<domain>.<tld>"),
        r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z]{2,})+$": ("FQDN", "<subdomain>.<domain>.<tld>"),
    }
    
    # Special handling for IP addresses
    ip_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
    if re.match(ip_pattern, value):
        if is_internal_ip(value):
            return "Internal IP", "<private_octet>.<private_octet>.<private_octet>.<private_octet>"
        else:
            return "External IP", "<public_octet>.<public_octet>.<public_octet>.<public_octet>"

    # Check other known patterns
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

def is_valid_match(term, text):
    """
    Check if a term appears in text with proper word boundaries.
    Matches only if the term is:
    1. The entire string
    2. At start followed by whitespace/non-word/underscore/digit
    3. At end preceded by whitespace/non-word/underscore/digit
    4. Between whitespace/non-word/underscore/digit characters
    
    Also matches non-space versions of terms that contain spaces.
    e.g., "Web Server" will match both "Web Server" and "WebServer"
    """
    # Escape special regex characters in the term
    escaped_term = re.escape(term)
    
    # If term contains spaces, create a pattern that matches both spaced and non-spaced versions
    if ' ' in term:
        # Create non-space version by removing spaces
        non_space_term = term.replace(' ', '')
        escaped_non_space = re.escape(non_space_term)
        
        # Build pattern for both versions using [\s\W_\d] for boundaries
        pattern = (
            f"(?i)("  # Start case-insensitive group
            f"(^{escaped_term}$)|(^{escaped_term}[\\s\\W_\\d])|([\\s\\W_\\d]+{escaped_term}$)|([\\s\\W_\\d]{escaped_term}[\\s\\W_\\d])|"  # Original spaced version
            f"(^{escaped_non_space}$)|(^{escaped_non_space}[\\s\\W_\\d])|([\\s\\W_\\d]+{escaped_non_space}$)|([\\s\\W_\\d]{escaped_non_space}[\\s\\W_\\d])"  # Non-spaced version
            f")"  # End group
        )
    else:
        # Original pattern for terms without spaces using [\s\W_\d] for boundaries
        pattern = f"(?i)(^{escaped_term}$)|(^{escaped_term}[\\s\\W_\\d])|([\\s\\W_\\d]+{escaped_term}$)|([\\s\\W_\\d]{escaped_term}[\\s\\W_\\d])"
    
    return re.search(pattern, text) is not None

def extract_country_region(text, *_):
    text_lower = str(text).lower()
    countries = set()
    regions = set()
    compliance = set()
    business_unit = set()

    # For each automaton, we'll now check for valid matches
    for _, (_, match) in COUNTRY_AUTOMATON.iter(text_lower):
        if is_valid_match(match.lower(), text_lower):
            countries.add(match)

    for _, (_, match) in REGION_AUTOMATON.iter(text_lower):
        if is_valid_match(match.lower(), text_lower):
            regions.add(match)

    for _, (_, match) in COMPLIANCE_AUTOMATON.iter(text_lower):
        if is_valid_match(match.lower(), text_lower):
            compliance.add(match)

    for _, (_, match) in BUSINESS_UNIT_AUTOMATON.iter(text_lower):
        if is_valid_match(match.lower(), text_lower):
            business_unit.add(match)
            
    return {
        "countries": list(countries),
        "regions": list(regions),
        "compliance": list(compliance),
        "business_unit": list(business_unit)
    }

def shorten_labels(labels, max_len=50): 
    """Shortens labels to a maximum length, reserving space for '...'."""
    return [label if len(label) <= max_len else label[:max_len - 3] + '...' for label in labels]

def create_bar_chart(df, x_col, y_col, title, color_scheme='blues'):
    """Helper function to create consistent bar charts with proper label handling"""
    # Shorten labels if needed
    df[y_col] = shorten_labels(df[y_col].astype(str))
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f'{x_col}:Q', title=x_col),
        y=alt.Y(f'{y_col}:N', 
                sort='-x', 
                title=y_col,
                axis=alt.Axis(labelLimit=0)),  # This ensures labels are not truncated
        color=alt.Color(f'{x_col}:Q', scale=alt.Scale(scheme=color_scheme))
    ).properties(
        width=600,
        height=max(400, len(df) * 25),  # Dynamic height based on number of bars
        title=title
    )
    return chart

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

st.markdown("## Load Previous Session")
uploaded_session = st.file_uploader("📂 Load Previous Session", type=["pkl"])
if uploaded_session:
    session_data = pickle.load(uploaded_session)
    df = session_data["dataframe"]
    primary_keys = session_data["primary_keys"]
    countries_input = session_data["countries_input"]
    regions_input = session_data["regions_input"]

    # Initialize custom categories in session state if not exists
    if "custom_categories" not in st.session_state:
        st.session_state.custom_categories = {}

    # Restore custom categories if present in session data
    if "custom_categories" in session_data:
        for cat, keywords in session_data["custom_categories"].items():
            # Only add if category doesn't exist or if it's different
            if cat not in st.session_state.custom_categories or st.session_state.custom_categories[cat]["keywords"] != keywords:
                automaton = build_automaton(keywords)
                st.session_state.custom_categories[cat] = {
                    "keywords": keywords,
                    "automaton": automaton
                }
                st.success(f"✅ Restored custom category: {cat}")

    st.success("✅ Session loaded successfully! Continue exploring below.")

# --- Dynamic Table of Contents ---
toc = """
# Table of Contents
- [Upload New File](#upload-new-file)
- [Field-wise Summary](#field-wise-summary)
- [Primary Key Identification](#primary-key-identification)
- [Per Field Insights](#per-field-insights)
- [Pattern Detection](#pattern-detection)
- [Country/Region Extraction Insights](#country-region-extraction-insights)
"""

# Append custom categories to the TOC if present
if "custom_categories" in st.session_state and st.session_state.custom_categories:
    toc += "\n- [Custom Extraction Insights](#custom-extraction-insights)"

st.sidebar.markdown(toc)

st.markdown("## Upload New File")
uploaded_file = st.file_uploader("Upload a CSV, Excel, JSON, or XML file", type=["csv", "xlsx", "json", "xml"])

# Toggle visibility of sidebar inputs using a checkbox
sidebar_visible = st.sidebar.checkbox("Show/Hide Custom Extraction Configs", value=True)

if sidebar_visible:
    # Sidebar inputs (only visible if the checkbox is checked)
    st.sidebar.header("Custom Extraction Configs")

    countries_input = st.sidebar.text_area(
        "Country List (comma separated)",
        value="India, Bharat, Republic of India, United Arab Emirates, UAE, Emirates, Saudi Arabia, KSA, Kingdom of Saudi Arabia, United Kingdom, UK, Britain, Great Britain, United States of America, USA, US, United States, America, Armenia, Republic of Armenia, Azerbaijan, Republic of Azerbaijan, Canada, Côte d'Ivoire, Ivory Coast, Chile, Republic of Chile, Colombia, Republic of Colombia, Costa Rica, Republic of Costa Rica, Germany, Deutschland, Federal Republic of Germany, Ecuador, Republic of Ecuador, Egypt, Arab Republic of Egypt, Spain, España, Kingdom of Spain, France, French Republic, Georgia, Sakartvelo, Ghana, Republic of Ghana, Croatia, Republic of Croatia, Italy, Italian Republic, Japan, Nippon, Nihon, Republic of Korea, South Korea, Korea (South), Lithuania, Republic of Lithuania, Luxembourg, Grand Duchy of Luxembourg, Morocco, Kingdom of Morocco, TFYR Macedonia, North Macedonia, Macedonia, Mexico, United Mexican States, Netherlands, Holland, Kingdom of the Netherlands, Philippines, Republic of the Philippines, Peru, Republic of Peru, Poland, Republic of Poland, Portugal, Portuguese Republic, Romania, Senegal, Republic of Senegal, Suriname, Republic of Suriname, Togo, Togolese Republic, Thailand, Kingdom of Thailand, Siam, Turkey, Türkiye, Republic of Turkey, Ethiopia, Federal Democratic Republic of Ethiopia, Algeria, People's Democratic Republic of Algeria, Jordan, Hashemite Kingdom of Jordan, Madagascar, Republic of Madagascar, Kazakhstan, Republic of Kazakhstan, China, People's Republic of China, PRC, Lebanon, Lebanese Republic, Serbia, Republic of Serbia, South Africa, Republic of South Africa, United Republic of Tanzania, Tanzania, Cameroon, Republic of Cameroon, Russian Federation, Russia, Switzerland, Swiss Confederation, Viet Nam, Vietnam, Socialist Republic of Vietnam, Nigeria, Federal Republic of Nigeria, Indonesia, Republic of Indonesia, Uganda, Republic of Uganda, Ukraine, Rwanda, Republic of Rwanda, Gabon, Gabonese Republic, Belarus, Kenya, Republic of Kenya, Kosovo, Republic of Kosovo, Tunisia, Republic of Tunisia, Uzbekistan, Republic of Uzbekistan, Albania, Republic of Albania, Jamaica, CTSS, Argentina, Argentine Republic, Australia, Commonwealth of Australia, Bosnia and Herzegovina, BiH, Belgium, Kingdom of Belgium, Brazil, Federative Republic of Brazil, Czech Republic, Czechia, Denmark, Kingdom of Denmark, Dominican Republic, Finland, Republic of Finland, Greece, Hellenic Republic, Mauritius, Republic of Mauritius, Guatemala, Republic of Guatemala, Guyana, Co-operative Republic of Guyana, Honduras, Republic of Honduras, Ireland, Éire, Republic of Ireland, Malaysia, Nicaragua, Republic of Nicaragua, Norway, Kingdom of Norway, Sweden, Kingdom of Sweden, Singapore, Republic of Singapore, El Salvador, Republic of El Salvador, Estonia, Republic of Estonia"
    )
    
    regions_input = st.sidebar.text_area(
        "Region List (comma separated)",
        value="APAC, EMEA, EWAP, Global, INDIA, LATAM, MAJOREL, Specialized Services, TGI"
    )

    compliance_input = st.sidebar.text_area(
        "Compliance List (comma separated)",
        value="GDPR, CCPA, HIPAA, PCI, PCI DSS, ISO 27001, SOC 2, NIST, FISMA, GLBA, SOX, FedRAMP, CMMC, NIST 800-53, NIST 800-171, ISO 27701, ISO 22301, ISO 31000, ISO 9001, ISO 14001, ISO 45001, ISO 20000, ISO 27017, ISO 27018, ISO 27002, ISO 27005, ISO 27006, ISO 27007, ISO 27008, ISO 27009, ISO 27010, ISO 27011, ISO 27012, ISO 27013, ISO 27014, ISO 27015, ISO 27016, ISO 27019, ISO 27020, ISO 27021, ISO 27022, ISO 27023, ISO 27024, ISO 27025, ISO 27026, ISO 27027, ISO 27028, ISO 27029, ISO 27030, ISO 27031, ISO 27032, ISO 27033, ISO 27034, ISO 27035, ISO 27036, ISO 27037, ISO 27038, ISO 27039, ISO 27040, ISO 27041, ISO 27042, ISO 27043, ISO 27044, ISO 27045, ISO 27046, ISO 27047, ISO 27048, ISO 27049, ISO 27050, ISO 27051, ISO 27052, ISO 27053, ISO 27054, ISO 27055, ISO 27056, ISO 27057, ISO 27058, ISO 27059, ISO 27060, ISO 27061, ISO 27062, ISO 27063, ISO 27064, ISO 27065, ISO 27066, ISO 27067, ISO 27068, ISO 27069, ISO 27070, ISO 27071, ISO 27072, ISO 27073, ISO 27074, ISO 27075, ISO 27076, ISO 27077, ISO 27078, ISO 27079, ISO 27080, ISO 27081, ISO 27082, ISO 27083, ISO 27084, ISO 27085, ISO 27086, ISO 27087, ISO 27088, ISO 27089, ISO 27090, ISO 27091, ISO 27092, ISO 27093, ISO 27094, ISO 27095, ISO 27096, ISO 27097, ISO 27098, ISO 27099, ISO 27100"
    )

    business_unit_input = st.sidebar.text_area(
        "Business Unit List (comma separated)",
        value="IT-SOFTWARE, CLIENT OPERATIONS, WORKFORCE, STAFF, CS-CLIENT SERVICES, BUSINESS DEVELOPMENT, OPS-CLIENT DELIVERY, HR, Client Operations, Workforce Management, WORKFORCE MANAGEMENT, Facilities, SALES, LC-AUDIT, FA-FP&A, HR-RECRUITMENT / TALENT ACQUISITION, Ops-Client Delivery, HR, MANAGEMENT/MANAGERS, Client Services, OPS-WORKFORCE MANAGEMENT, LEGAL, AF-PREMISES AND ADMINISTRATION, SUPPORT HELP DESK, HR-CROSS FUNCTION ROLES, Infrastructure Desktop, Quality Assurance, OPS-GLOBAL PROCESSES, STANDARDS AND CONTINUOUS IMPROVEMENT, SUPPORT ANALYST, DEVELOPMENT DEVELOPER, IT-INFRASTRUCTURE OPERATIONS, IT-INFORMATION SECURITY, Training, MANAGEMENT/MANAGERS, CLIENT, SUPPORT ADMINISTRATION, HR-TRAINING, IT, HUMAN RESOURCES, BUSINESS INTELLIGENCE, HR-PAYROLL, HR-LEARNING AND DEVELOPMENT AND ORGANIZATIONAL DEVELOPMENT, INFRASTRUCTURE DESKTOP, OPS-BUSINESS INTELLIGENCE AND REPORTING, TRAINING, RECRUITING, OPS-QUALITY ANALYSIS / CONTINUOUS IMPROVEMENT, STAFF, STAFF, OPS-CROSS FUNCTION ROLES, IT-SERVICE DESK AND IT SERVICE MANAGEMENT, QUALITY ASSURANCE, FINANCE, IT-SUPPORT SERVICES, FA-PROCUREMENT AND SUPPORT, MKT- CROSS FUNCTION ROLES, IT-DATA COE, MANAGEMENT/MANAGERS, TRAINING, INFRASTRUCTURE SYSTEMS, Support Help Desk, MARKETING, HR-ONBOARDING, AF-HEALTH AND SAFETY, Professional Services, FACILITIES, AF-MAINTENANCE, ADMINISTRATION, DEVELOPMENT SCRIPTING, BD-BUSINESS DEVELOPMENT, HR-Training, CLIENT SERVICES, CS-STRATEGIC ACCOUNT MANAGEMENT, IT-INFRASTRUCTURE ARCHITECTURE AND ENGINEERING, MANAGEMENT/MANAGERS, HR, SECURITY, PAYROLL, MANAGEMENT TRAINING DEVELOPMENT, ANALYST, DS-CROSS FUNCTION ROLES, RISK, Ops-Quality Analysis / Continuous Improvement, Finance, OPS-PROJECT MANAGEMENT, OPS-Cross Function Roles, PROCUREMENT, INFORMATION SECURITY, HR-EMPLOYEE RELATIONS, EXECUTIVE MANAGEMENT, FA-CROSS FUNCTION ROLES, LC-COMPLIANCE, IT-Service Desk and IT Service Management, BD-CROSS FUNCTION ROLES, APPLICATION SUPPORT, Ops-Workforce Management, STDS-ROLLOUT & AUDIT, DS-CONSULTING AND SOLUTIONING, IT-Infrastructure Operations, CS-Client Services, IT-Software, HR-HELP DESK, INFRASTRUCTURE TELECOM, HR-COMPENSATION AND BENEFITS, MKT-DIGITAL MARKETING, INFRASTRUCTURE NETWORK, LC-LEGAL, Business Development, HR-Cross Function Roles, FA-Cross Function Roles, FA-FINANCIAL SYSTEMS, TRANSFORMATION AND AUTOMATION, BD-SALES ENABLEMENT, MKT-WEB DESIGN, HR-Payroll, Marketing, IT-CROSS FUNCTION ROLES, BD-PRESALES, INFRASTRUCTURE DATABASE, Recruiting, BD-Business Development, SALES, CLIENT, EM-EXECUTIVE ASSISTANTS, IT, STAFF, Development Scripting, TRANSFORMATION, EXTERNAL, IT, MANAGEMENT/MANAGERS, HR-DIVERSITY, EQUITY AND INCLUSION, Management Training Development, CORPORATE COMPLIANCE, Infrastructure Database, Human Resources, SUPPORT PRODUCT MANAGEMENT, AF-CROSS FUNCTION ROLES, DS-DATA ANALYTICS, Transformation, SUPPORT PROJECT MANAGEMENT, Risk, STDS-IMPROVEMENT DELIVERY PERFORMANCE, STAFF, TRAINING, IT-Infrastructure Architecture and Engineering, Infrastructure Systems, STRATEGIC ACCOUNT MANAGEMENT, IT-PLATFORM ENGINEERING SOLUTIONS COE, BUSINESS OPERATIONS, AF-LOGISTICS AND WAREHOUSING, DS-Consulting and Solutioning, MKT-CONTENT DEVELOPMENT, EM-LOCAL EXECUTIVE MANAGEMENT TIER 3, STAFF, STAFF, STAFF, Ops-Business Intelligence and Reporting, Legal, EM-LOCAL EXECUTIVE MANAGEMENT, FA-TREASURY, STAFF, IT, Support Product Management, LC-DATA PRIVACY, AUDITOR, Business Operations, Ops-Project Management, Support Project Management, HR, MANAGEMENT/MANAGERS, TRAINING, Business Intelligence, EM-LOCAL EXECUTIVE MANAGEMENT TIER 1, DEVELOPMENT ARCHITECT, AF-Premises and Administration, TRAINING, MANAGEMENT/MANAGERS, Payroll, Procurement, AF-Maintenance, Ops-Global Processes, Standards and Continuous Improvement, IT, STAFF, STAFF, HR-Onboarding, IT, TRAINING, HR-Learning and Development and Organizational Development"
    )

    COUNTRY_LIST = [x.strip() for x in countries_input.split(",")]
    REGION_LIST = [x.strip() for x in regions_input.split(",")]
    COMPLIANCE_LIST = [x.strip() for x in compliance_input.split(",")]
    BUSINESS_UNIT_LIST = [x.strip() for x in business_unit_input.split(",")]

    # Build automatons once
    COUNTRY_AUTOMATON = build_automaton(COUNTRY_LIST)
    REGION_AUTOMATON = build_automaton(REGION_LIST)
    COMPLIANCE_AUTOMATON = build_automaton(COMPLIANCE_LIST)
    BUSINESS_UNIT_AUTOMATON = build_automaton(BUSINESS_UNIT_LIST)

    with st.sidebar.expander("➕ Add Custom Extraction Categories"):
        with st.form(key="custom_extraction_form"):
            custom_category = st.text_input("Category Name (e.g., Product, Company)", key="category_input")
            custom_keywords_input = st.text_area("Keywords (comma separated)", key="keywords_input")
            submitted = st.form_submit_button("Add Custom Category")

            if submitted:
                if custom_category and custom_keywords_input:
                    # Initialize custom categories if not exists
                    if "custom_categories" not in st.session_state:
                        st.session_state.custom_categories = {}

                    # Prevent duplicate category overwrite unless intentional
                    if custom_category in st.session_state.custom_categories:
                        st.warning(f"⚠️ Category '{custom_category}' already exists. Choose another name.")
                    else:
                        keywords = [kw.strip() for kw in custom_keywords_input.split(",") if kw.strip()]
                        automaton = build_automaton(keywords)
                        st.session_state.custom_categories[custom_category] = {
                            "keywords": keywords,
                            "automaton": automaton
                        }
                        st.success(f"✅ Category '{custom_category}' added with {len(keywords)} keywords.")
                        # Force a rerun to ensure the UI updates
                        st.rerun()
                else:
                    st.error("❌ Please enter both a category name and at least one keyword.")

    if "custom_categories" in st.session_state and st.session_state.custom_categories:
        st.markdown("### 🗂️ Current Custom Categories")

        for cat_name, meta in st.session_state.custom_categories.items():
            with st.expander(f"🔧 `{cat_name}` ({len(meta['keywords'])} keywords)"):
                st.write(", ".join(meta["keywords"]))

                # Optional: edit or delete
                new_keywords = st.text_area(f"✏️ Edit keywords for `{cat_name}`", value=", ".join(meta["keywords"]), key=f"edit_{cat_name}")
                if st.button(f"Update `{cat_name}`", key=f"update_{cat_name}"):
                    new_kw_list = [kw.strip() for kw in new_keywords.split(",") if kw.strip()]
                    st.session_state.custom_categories[cat_name]["keywords"] = new_kw_list
                    st.session_state.custom_categories[cat_name]["automaton"] = build_automaton(new_kw_list)
                    st.success(f"✅ Updated keywords for `{cat_name}`")

                if st.button(f"❌ Delete `{cat_name}`", key=f"delete_{cat_name}"):
                    del st.session_state.custom_categories[cat_name]
                    st.warning(f"🗑️ Deleted category `{cat_name}`")
                    st.rerun()


else:
    st.sidebar.write("Sidebar content is hidden.")

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        try:
            # Try to read as JSON first
            df = pd.read_json(uploaded_file)
        except Exception as e:
            # If that fails, try to read as JSON lines
            try:
                df = pd.read_json(uploaded_file, lines=True)
            except Exception as e2:
                st.error(f"Error reading JSON file: {str(e2)}")
                st.info("Please ensure your JSON file is either a valid JSON array of objects or JSON Lines format.")
                df = None
    elif uploaded_file.name.endswith(".xml"):
        try:
            # Read the XML file content
            xml_content = uploaded_file.read()
            
            # Try to parse as XML
            try:
                # First attempt: Try to parse as a simple XML structure
                root = etree.fromstring(xml_content)
                
                # Get all elements at the first level that have children
                records = []
                for elem in root:
                    if len(elem) > 0:  # Element has children
                        record = {}
                        for child in elem:
                            record[child.tag] = child.text
                        records.append(record)
                
                if records:
                    df = pd.DataFrame(records)
                else:
                    # If no nested structure found, try to parse as a flat structure
                    records = []
                    for elem in root:
                        record = {elem.tag: elem.text}
                        records.append(record)
                    df = pd.DataFrame(records)
                    
            except Exception as e:
                # Second attempt: Try using xmltodict for more complex XML structures
                try:
                    # Convert XML to dict
                    xml_dict = xmltodict.parse(xml_content)
                    
                    # Function to flatten nested dictionary
                    def flatten_dict(d, parent_key='', sep='_'):
                        items = []
                        for k, v in d.items():
                            new_key = f"{parent_key}{sep}{k}" if parent_key else k
                            if isinstance(v, dict):
                                items.extend(flatten_dict(v, new_key, sep=sep).items())
                            elif isinstance(v, list):
                                # Handle lists by creating separate records
                                for i, item in enumerate(v):
                                    if isinstance(item, dict):
                                        items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                                    else:
                                        items.append((f"{new_key}_{i}", item))
                            else:
                                items.append((new_key, v))
                        return dict(items)
                    
                    # Flatten the dictionary
                    flat_dict = flatten_dict(xml_dict)
                    
                    # Convert to DataFrame
                    if isinstance(flat_dict, dict):
                        df = pd.DataFrame([flat_dict])
                    else:
                        df = pd.DataFrame(flat_dict)
                        
                except Exception as e2:
                    st.error(f"Error reading XML file: {str(e2)}")
                    st.info("""
                    Please ensure your XML file is in one of these formats:
                    1. Simple XML with repeating elements (e.g., <root><item><field>value</field></item></root>)
                    2. Complex XML with nested structures (will be flattened)
                    """)
                    df = None
                    
        except Exception as e:
            st.error(f"Error processing XML file: {str(e)}")
            df = None
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"✅ Loaded **{df.shape[0]}** records with **{df.shape[1]}** fields.")
elif "df" in locals():
    st.info("ℹ️ Using data loaded from session file.")
else:
    df = None
    st.info("📂 Please upload a file to begin analysis.")

if df is not None:
    # Add Data Filtering Section in sidebar
    if sidebar_visible:
        st.sidebar.markdown("---")  # Add a separator
        st.sidebar.markdown("### 🔍 Data Filters")
        
        # Initialize session state for filters if not exists
        if 'active_filters' not in st.session_state:
            st.session_state.active_filters = {}
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None
        
        # Create a form for filter selections
        with st.sidebar.form(key="filter_form"):
            st.markdown("### 🔍 Data Filters")
            
            # Field selection for filtering
            filter_fields = st.multiselect(
                "Select fields to filter on",
                options=df.columns.tolist(),
                key="filter_fields"
            )
            
            # Dictionary to store filter selections
            filter_selections = {}
            
            # Create filter widgets for each selected field
            for field in filter_fields:
                try:
                    # Get available values from the original dataframe
                    non_null_mask = df[field].notna()
                    available_values = df.loc[non_null_mask, field].astype(str)
                    available_values = available_values[~available_values.str.lower().isin(['nan', 'none', ''])]
                    available_values = sorted(available_values.unique(), key=lambda x: str(x).lower())
                    
                    # Get current selected values for this field
                    current_selected = [v for v in st.session_state.active_filters.get(field, []) if str(v) in available_values]
                    
                    # Create a multi-select widget for each field
                    selected_values = st.multiselect(
                        f"Filter {field}",
                        options=available_values,
                        default=current_selected,
                        key=f"filter_{field}"
                    )
                    
                    if selected_values:
                        filter_selections[field] = selected_values
                        
                except Exception as e:
                    st.error(f"Error processing field {field}: {str(e)}")
                    continue
            
            # Add form submit and clear buttons
            col1, col2 = st.columns(2)
            with col1:
                apply_filters = st.form_submit_button("Apply Filters")
            with col2:
                clear_filters = st.form_submit_button("Clear All Filters")
        
        # Handle form submission
        if apply_filters:
            if filter_selections:
                filtered_df = df.copy()
                for field, values in filter_selections.items():
                    try:
                        non_null_mask = filtered_df[field].notna()
                        filtered_df.loc[non_null_mask, field] = filtered_df.loc[non_null_mask, field].astype(str)
                        mask = filtered_df[field].isin([str(v) for v in values])
                        filtered_df = filtered_df[mask]
                    except Exception as e:
                        st.error(f"Error filtering field {field}: {str(e)}")
                        continue
                
                # Update the filtered data and active filters
                st.session_state.filtered_data = filtered_df
                st.session_state.active_filters = filter_selections
            else:
                st.session_state.filtered_data = None
                st.session_state.active_filters = {}
            st.rerun()
        
        elif clear_filters:
            st.session_state.active_filters = {}
            st.session_state.filtered_data = None
            st.rerun()
        
        # Show active filters if any
        if st.session_state.active_filters:
            st.sidebar.markdown("---")
            st.sidebar.markdown("#### Active Filters:")
            for field, values in st.session_state.active_filters.items():
                try:
                    display_values = []
                    for v in values:
                        try:
                            if pd.isna(v):
                                continue
                            display_values.append(str(v))
                        except:
                            continue
                    if display_values:
                        st.sidebar.markdown(f"- **{field}**: {', '.join(display_values)}")
                except Exception as e:
                    st.sidebar.error(f"Error displaying filter for {field}")
        
        # Use filtered data if available
        if st.session_state.filtered_data is not None:
            df = st.session_state.filtered_data

    st.markdown("## Field-wise Summary")
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

    st.markdown("## Primary Key Identification")
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


    st.markdown("## Per Field Insights")
    st.subheader("📌 Per Field Insights")
    for col in df.columns:
        st.markdown(f"### 🧬 {col}")
        col_data = df[col].dropna()
        total = len(df)
        # Calculate coverage percentage safely
        coverage = 100 - (df[col].isnull().sum() / len(df) * 100)
        # Ensure coverage is a valid number between 0 and 100
        coverage = max(0, min(100, float(coverage) if pd.notna(coverage) else 0))
        
        # Use the safe coverage value for the progress bar
        st.progress(coverage/100, text=f"Coverage: {coverage:.2f}%")
        nunique = col_data.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        if nunique == total:
            with st.expander("📋 View Unique Values (with counts)"):
                value_counts = col_data.value_counts().reset_index()
                value_counts.columns = ["Value", "Count"]
                value_counts = value_counts.sort_values("Count", ascending=False)
                st.dataframe(value_counts, use_container_width=True)
        elif not is_numeric:
            try:
                # Try parsing as datetime
                parsed_col = pd.to_datetime(col_data, errors="coerce", infer_datetime_format=True)
                if parsed_col.notna().sum() == 0 and pd.api.types.is_numeric_dtype(col_data):
                    parsed_col = pd.to_datetime(col_data, errors="coerce", unit="ms")
                    if parsed_col.notna().sum() == 0:
                        parsed_col = pd.to_datetime(col_data, errors="coerce", unit="s")

                if parsed_col.notna().sum() > 0:
                    st.markdown("#### 📈 Trend (Date/Time Field)")

                    temp_df = df.copy()
                    temp_df['__datetime__'] = parsed_col
                    temp_df = temp_df.dropna(subset=['__datetime__'])

                    # UI: frequency selection
                    freq = st.selectbox(
                        f"Choose trend resolution for `{col}`",
                        options=["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                        index=0,
                        key=f"freq_{col}"
                    )

                    freq_map = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "M",
                        "Quarterly": "Q",
                        "Yearly": "Y"
                    }

                    # Add checkbox to toggle between all dates and date range
                    show_all_dates = st.checkbox(
                        f"Show all dates for `{col}`",
                        value=False,
                        key=f"show_all_{col}"
                    )

                    if not show_all_dates:
                        # UI: date range picker (only show if not showing all dates)
                        min_date = parsed_col.min().date()
                        max_date = parsed_col.max().date()
                        start_date, end_date = st.date_input(
                            f"Select date range for `{col}`",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key=f"range_{col}"
                        )

                        # Filter by date range
                        filtered_df = temp_df[
                            (temp_df['__datetime__'].dt.date >= start_date) &
                            (temp_df['__datetime__'].dt.date <= end_date)
                        ].copy()
                    else:
                        # Use all dates
                        filtered_df = temp_df.copy()

                    resampled = filtered_df.set_index("__datetime__").resample(freq_map[freq])

                    record_counts = resampled.size().rename("Total Records")
                    chart_df = pd.DataFrame(record_counts)

                    if primary_keys:
                        unique_keys_df = filtered_df.drop_duplicates(subset=primary_keys)
                        unique_counts = unique_keys_df.set_index("__datetime__").resample(freq_map[freq]).size().rename("Unique Primary Keys")
                        chart_df = chart_df.join(unique_counts, how='outer').fillna(0)

                    st.line_chart(chart_df)
                    continue  # Skip to next column
            except Exception:
                pass

            avg_str_len = col_data.astype(str).apply(len).mean()

            if avg_str_len > 50:
                st.markdown("#### ☁️ Word Cloud (Long Text Field)")

                # Pre-filter to only include meaningful values
                filtered_values = col_data.dropna().astype(str).tolist()
                filtered_values = [val for val in filtered_values if any(c.isalpha() for c in val)]
                text = ' '.join(filtered_values).strip()

                try:
                    if text:
                        wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA').generate(text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        fig.patch.set_alpha(0)  # 🔥 Removes the white border
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        raise ValueError("No valid words")
                except Exception as e:
                    st.warning("⚠️ Word cloud not generated due to non-textual content. Falling back to bar chart.")

                    # Fall back to bar chart
                    top_n = 10
                    val_counts_total = df[col].value_counts().head(top_n)
                    percent_total = (val_counts_total / total * 100).round(2)

                    chart_df = pd.DataFrame({
                        col: val_counts_total.index.tolist(),
                        'Occurrences': val_counts_total.values
                    })

                    # Use the new helper function for consistent bar charts
                    chart = create_bar_chart(chart_df, 'Occurrences', col, "Top 10 Values (All Records)")
                    st.altair_chart(chart, use_container_width=True)
            else:
                top_n = 10

                st.markdown("#### Top Values (All Rows)")
                val_counts_total = df[col].value_counts().head(top_n)
                percent_total = (val_counts_total / total * 100).round(2)

                # Create a DataFrame for Altair
                chart_df = pd.DataFrame({
                    col: val_counts_total.index.tolist(),
                    'Occurrences': val_counts_total.values
                })

                # Use the new helper function for consistent bar charts
                chart = create_bar_chart(chart_df, 'Occurrences', col, "Top 10 Values (All Records)")
                st.altair_chart(chart, use_container_width=True)

            # Create a table showing values, counts, and percentages
            percent_table = pd.DataFrame({
                "Value": val_counts_total.index,
                "Count": val_counts_total.values,
                "Percentage (%)": percent_total
            })

            # Add total count to the table
            percent_table = pd.concat([percent_table, pd.DataFrame([["Total", total, "100"]], columns=percent_table.columns)], ignore_index=True)

            percent_table["Percentage (%)"] = percent_table["Percentage (%)"].astype(float)

            st.markdown("### 📊 Value Counts and Percentages")
            st.dataframe(percent_table, use_container_width=True)

            # Determine primary key(s)
            primary_keys = manual_keys if 'manual_keys' in locals() and manual_keys else auto_keys if 'auto_keys' in locals() and auto_keys else []

            if primary_keys:
                try:
                    temp_df = df[primary_keys + [col]].dropna().drop_duplicates(subset=primary_keys)
                    grouped_counts = temp_df[col].value_counts().head(top_n)
                    percent_keys = (grouped_counts / temp_df.shape[0] * 100).round(2)

                    # Make sure grouped_counts is already computed correctly
                    chart_df2 = pd.DataFrame({
                        col: grouped_counts.index.tolist(),
                        'Occurrences': grouped_counts.values
                    })

                    # Use the new helper function for consistent bar charts
                    chart2 = create_bar_chart(chart_df2, 'Occurrences', col, "Top 10 Values (Per Unique Primary Key)", color_scheme='greens')
                    st.markdown("#### Top Values (Per Primary Key)")
                    st.altair_chart(chart2, use_container_width=True)


                    # Create a table showing per primary key values, counts, and percentages
                    percent_key_table = pd.DataFrame({
                        "Value": grouped_counts.index,
                        "Count": grouped_counts.values,
                        "Percentage (%)": percent_keys
                    })

                    # Add total count to the table
                    percent_key_table = pd.concat([percent_key_table, pd.DataFrame([["Total", temp_df.shape[0], "100"]], columns=percent_key_table.columns)], ignore_index=True)

                    percent_key_table["Percentage (%)"] = percent_table["Percentage (%)"].astype(float)

                    st.markdown("### 📊 Value Counts and Percentages (Per Primary Key)")
                    st.dataframe(percent_key_table, use_container_width=True)

                except Exception as e:
                    st.error(f"⚠️ Error generating primary key-based chart: {e}")
            else:
                st.info("ℹ️ No primary key selected or detected, so primary key based chart is skipped.")
        else:
            chart_df = pd.DataFrame({col: col_data})

            # Create histogram with Altair
            hist = alt.Chart(chart_df).mark_bar(color='teal').encode(
                alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col),
                y=alt.Y('count()', title='Count')
            ).properties(
                width=600,
                height=300,
                title="Distribution"
            )

            st.altair_chart(hist, use_container_width=True)


    st.markdown("## Pattern Detection")
    st.subheader("🔍 Pattern Detection")
    st.markdown("""
    Each value is scanned for **known formats** like:
    - **Internal IP**: Private IP addresses (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, etc.)
    - **External IP**: Public IP addresses
    - **MAC Address**: Network hardware addresses
    - **Email**: Email addresses
    - **FQDN**: Fully Qualified Domain Names
    
    If not matched, a symbolic abstraction is used:
    - **A** = Uppercase letter
    - **a** = Lowercase letter
    - **9** = Digit
    - **@** = Special character
    - **?** = Other
    """)

    all_pattern_info = []

    for col in df.columns:
        try:
            col_data = df[col].dropna().astype(str)
            if len(col_data) == 0:  # Skip if no data after filtering
                continue
                
            patterns = col_data.apply(detect_pattern)

            # Count pattern frequencies
            pattern_counts = patterns.apply(lambda x: x[0]).value_counts()
            total = pattern_counts.sum()

            if total == 0:  # Skip if no patterns found
                continue

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
        except Exception as e:
            st.error(f"Error processing patterns for field {col}: {str(e)}")
            continue

    pattern_df = pd.DataFrame(all_pattern_info)

    if not pattern_df.empty:
        st.markdown("### 📋 Detailed Pattern Report")
        st.dataframe(pattern_df, use_container_width=True)

        st.markdown("### 🌟 Most Common Pattern Per Field")
        try:
            top_patterns = pattern_df.sort_values('Confidence (%)', ascending=False).drop_duplicates('Field')
            st.dataframe(top_patterns[['Field', 'Pattern', 'Example', 'Confidence (%)']], use_container_width=True)
        except Exception as e:
            st.error("Error generating most common patterns. No patterns found in the filtered data.")
    else:
        st.info("No patterns detected in the current data. Try adjusting your filters or check if the data contains any patterns.")

    with st.expander("📤 Export Pattern Detection Results"):
        if not pattern_df.empty:
            csv = pattern_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", data=csv, file_name="all_patterns.csv", mime="text/csv")

            html = pattern_df.to_html(index=False)
            st.download_button("📄 Download HTML", data=html, file_name="all_patterns.html", mime="text/html")
        else:
            st.info("No patterns to export.")

    if sidebar_visible:
        # Assuming extract_country_region and shorten_labels functions are defined elsewhere.

        st.markdown("## Country/Region Extraction Insights")
        st.subheader("🌍 Country/Region Extraction Insights")
        summary_data = []
        region_summary_data = []

        # Get the total number of records in the DataFrame
        total_records = len(df)

        for col in df.select_dtypes(include=['object', 'string']).columns:
            non_null_values = df[col].dropna()
            sampled_values = non_null_values  # No sampling
            records_processed = len(sampled_values)

            # Apply extraction
            results = sampled_values.apply(lambda x: (x, extract_country_region(x)))

            country_counts = {}
            country_evidence = {}

            region_counts = {}
            region_evidence = {}

            for val, res in results:
                for c in res['countries']:
                    country_counts[c] = country_counts.get(c, 0) + 1
                    if c not in country_evidence:
                        country_evidence[c] = val  # first evidence sample

                for c in res['regions']:
                    region_counts[c] = region_counts.get(c, 0) + 1
                    if c not in region_evidence:
                        region_evidence[c] = val  # first evidence sample

            # Create country summary if countries found
            if country_counts:
                # Count records that have at least one country
                records_with_countries = sum(1 for _, res in results if res['countries'])
                coverage_percentage = (records_with_countries / total_records) * 100

                summary_data.append({
                    'Field': col,
                    'Countries Found': ', '.join(sorted(country_counts.keys())),
                    'Coverage': f"{records_with_countries} ({coverage_percentage:.2f}%)",
                    'Evidence': [country_evidence[c] for c in sorted(country_counts.keys())],
                    'Records Processed': records_processed
                })

            # Create region summary if regions found
            if region_counts:
                # Count records that have at least one region
                records_with_regions = sum(1 for _, res in results if res['regions'])
                coverage_percentage = (records_with_regions / total_records) * 100

                region_summary_data.append({
                    'Field': col,
                    'Regions Found': ', '.join(sorted(region_counts.keys())),
                    'Coverage': f"{records_with_regions} ({coverage_percentage:.2f}%)",
                    'Evidence': [region_evidence[c] for c in sorted(region_counts.keys())],
                    'Records Processed': records_processed
                })

        # Create DataFrames for both summaries
        summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
        region_summary_df = pd.DataFrame(region_summary_data) if region_summary_data else pd.DataFrame()

        # Show both summaries
        if not summary_df.empty:
            st.write("### 🌍 Country Extraction Summary by Column")
            st.dataframe(summary_df)
            with st.expander("📤 Export Country Results"):
                csv = summary_df.to_csv(index=False).encode()
                st.download_button(
                    "📄 Download as CSV",
                    data=csv,
                    file_name="country_extraction.csv",
                    mime="text/csv"
                )
        else:
            st.write("No countries were extracted from the data.")

        if not region_summary_df.empty:
            st.write("### 🌐 Region Extraction Summary by Column")
            st.dataframe(region_summary_df)
            with st.expander("📤 Export Region Results"):
                csv = region_summary_df.to_csv(index=False).encode()
                st.download_button(
                    "📄 Download as CSV",
                    data=csv,
                    file_name="region_extraction.csv",
                    mime="text/csv"
                )
        else:
            st.write("No regions were extracted from the data.")

        # Add Compliance Summary
        compliance_summary_data = []
        for col in df.select_dtypes(include=['object', 'string']).columns:
            non_null_values = df[col].dropna()
            sampled_values = non_null_values
            records_processed = len(sampled_values)
            
            results = sampled_values.apply(lambda x: (x, extract_country_region(x)))
            
            compliance_counts = {}
            compliance_evidence = {}
            
            for val, res in results:
                for c in res['compliance']:
                    compliance_counts[c] = compliance_counts.get(c, 0) + 1
                    if c not in compliance_evidence:
                        compliance_evidence[c] = val

            if compliance_counts:
                records_with_compliance = sum(1 for _, res in results if res['compliance'])
                coverage_percentage = (records_with_compliance / total_records) * 100

                compliance_summary_data.append({
                    'Field': col,
                    'Compliance Found': ', '.join(sorted(compliance_counts.keys())),
                    'Coverage': f"{records_with_compliance} ({coverage_percentage:.2f}%)",
                    'Evidence': [compliance_evidence[c] for c in sorted(compliance_counts.keys())],
                    'Records Processed': records_processed
                })

        compliance_summary_df = pd.DataFrame(compliance_summary_data) if compliance_summary_data else pd.DataFrame()

        if not compliance_summary_df.empty:
            st.write("### 📋 Compliance Extraction Summary by Column")
            st.dataframe(compliance_summary_df)
            with st.expander("📤 Export Compliance Results"):
                csv = compliance_summary_df.to_csv(index=False).encode()
                st.download_button(
                    "📄 Download as CSV",
                    data=csv,
                    file_name="compliance_extraction.csv",
                    mime="text/csv"
                )
        else:
            st.write("No compliance terms were extracted from the data.")

        # Add Business Unit Summary
        business_unit_summary_data = []
        for col in df.select_dtypes(include=['object', 'string']).columns:
            non_null_values = df[col].dropna()
            sampled_values = non_null_values
            records_processed = len(sampled_values)
            
            results = sampled_values.apply(lambda x: (x, extract_country_region(x)))
            
            business_unit_counts = {}
            business_unit_evidence = {}
            
            for val, res in results:
                for c in res['business_unit']:
                    business_unit_counts[c] = business_unit_counts.get(c, 0) + 1
                    if c not in business_unit_evidence:
                        business_unit_evidence[c] = val

            if business_unit_counts:
                records_with_business_unit = sum(1 for _, res in results if res['business_unit'])
                coverage_percentage = (records_with_business_unit / total_records) * 100

                business_unit_summary_data.append({
                    'Field': col,
                    'Business Units Found': ', '.join(sorted(business_unit_counts.keys())),
                    'Coverage': f"{records_with_business_unit} ({coverage_percentage:.2f}%)",
                    'Evidence': [business_unit_evidence[c] for c in sorted(business_unit_counts.keys())],
                    'Records Processed': records_processed
                })

        business_unit_summary_df = pd.DataFrame(business_unit_summary_data) if business_unit_summary_data else pd.DataFrame()

        if not business_unit_summary_df.empty:
            st.write("### 🏢 Business Unit Extraction Summary by Column")
            st.dataframe(business_unit_summary_df)
            with st.expander("📤 Export Business Unit Results"):
                csv = business_unit_summary_df.to_csv(index=False).encode()
                st.download_button(
                    "📄 Download as CSV",
                    data=csv,
                    file_name="business_unit_extraction.csv",
                    mime="text/csv"
                )
        else:
            st.write("No business units were extracted from the data.")

    # --- Custom Extraction Summary (Structured like Country/Region) ---
    if "custom_categories" in st.session_state and df is not None:
        st.markdown("## 🧠 Custom Extraction Insights")
        
        # Add debug information about current custom categories
        with st.expander("🔧 Debug: Current Custom Categories"):
            st.write("Active custom categories:", list(st.session_state.custom_categories.keys()))
            for cat, meta in st.session_state.custom_categories.items():
                st.write(f"- {cat}: {len(meta['keywords'])} keywords")

        total_records = len(df)

        for category_name, meta in st.session_state.custom_categories.items():
            automaton = meta["automaton"]
            summary_data = []

            st.subheader(f"🔍 Extraction Summary for `{category_name}`")

            for col in df.select_dtypes(include=["object", "string"]).columns:
                non_null_values = df[col].dropna()
                sampled_values = non_null_values
                records_processed = len(sampled_values)

                # Update the extraction logic to use is_valid_match
                results = []
                for val in sampled_values:
                    val_lower = str(val).lower()
                    matches = set()
                    for _, (_, match) in automaton.iter(val_lower):
                        if is_valid_match(match.lower(), val_lower):
                            matches.add(match)
                    results.append((val, list(matches)))

                match_counts = {}
                match_evidence = {}

                for val, matches in results:
                    for m in matches:
                        match_counts[m] = match_counts.get(m, 0) + 1
                        if m not in match_evidence:
                            match_evidence[m] = val

                if match_counts:
                    total_mentions = sum(match_counts.values())
                    coverage_percentage = (total_mentions / total_records) * 100

                    summary_data.append({
                        "Field": col,
                        f"{category_name}s Found": ', '.join(sorted(match_counts.keys())),
                        "Coverage": f"{total_mentions} ({coverage_percentage:.2f}%)",
                        "Evidence": [match_evidence[m] for m in sorted(match_counts.keys())],
                        "Records Processed": records_processed
                    })

            summary_df = pd.DataFrame(summary_data)

            if not summary_df.empty:
                st.markdown(f"### Summary Table for `{category_name}`")
                st.dataframe(summary_df)
            else:
                st.info(f"No `{category_name}` matches found.")


if st.button("💾 Save Session"):
    if 'df' in locals() and df is not None:
        session_data = {
            "dataframe": df,
            "primary_keys": primary_keys,
            "countries_input": countries_input,
            "regions_input": regions_input,
            "custom_categories": {
                cat: data["keywords"] for cat, data in st.session_state.get("custom_categories", {}).items()
            }
        }

        # Prepare save directory
        save_dir = "EDA_Reports"
        os.makedirs(save_dir, exist_ok=True)

        # Build filename: <original_filename>_<timestamp>.pkl
        original_filename = uploaded_file.name.rsplit('.', 1)[0] if uploaded_file else "session"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{original_filename}_{timestamp}.pkl"
        save_path = os.path.join(save_dir, full_filename)

        # Save to file
        with open(save_path, "wb") as f:
            pickle.dump(session_data, f)

        st.success(f"✅ Session saved to `{save_path}`")
    else:
        st.warning("⚠️ No dataframe available to save.")
