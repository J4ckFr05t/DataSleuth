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
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from column_processor import process_single_column
import plotly.express as px
import plotly.graph_objects as go
import json
from pyhive import hive  # Add this import for Spark Thrift Server connection
import time  # Add time module for tracking analysis duration
from tqdm import tqdm

def process_patterns_parallel(col_data, col_name):
    """Process patterns for a single column in parallel"""
    try:
        patterns = col_data.apply(detect_pattern)
        pattern_counts = patterns.apply(lambda x: x[0]).value_counts()
        total = pattern_counts.sum()

        if total == 0:
            return []

        pattern_info = []
        for pat, count in pattern_counts.items():
            example = patterns[patterns.apply(lambda x: x[0]) == pat].iloc[0][1]
            confidence = round((count / total) * 100, 2)
            pattern_info.append({
                "Field": col_name,
                "Pattern": pat,
                "Example": example if example else "",
                "Confidence (%)": confidence
            })
        return pattern_info
    except Exception as e:
        st.error(f"Error processing patterns for field {col_name}: {str(e)}")
        return []

def process_extraction_parallel(col_data, col_name, total_records, extraction_func):
    """Process extractions (country/region/business unit) for a single column in parallel"""
    try:
        results = col_data.apply(lambda x: (x, extraction_func(x)))
        records_processed = len(col_data)
        
        # Initialize counters for each category
        counts = {
            'countries': {},
            'regions': {},
            'compliance': {},
            'business_unit': {}
        }
        evidence = {
            'countries': {},
            'regions': {},
            'compliance': {},
            'business_unit': {}
        }
        
        # Track unique records with matches for each category
        unique_records = {
            'countries': set(),
            'regions': set(),
            'compliance': set(),
            'business_unit': set()
        }
        
        # Process results
        for idx, (val, res) in enumerate(results):
            # Process each category explicitly
            for category in ['countries', 'regions', 'compliance', 'business_unit']:
                if category in res and res[category]:  # Check if category exists and has values
                    unique_records[category].add(idx)  # Add record index to unique set
                    for item in res[category]:
                        counts[category][item] = counts[category].get(item, 0) + 1
                        if item not in evidence[category]:
                            evidence[category][item] = val

        # Calculate coverage and create summary
        summary_data = []
        category_mapping = {
            'countries': 'Countries',
            'regions': 'Regions',
            'compliance': 'Compliance',
            'business_unit': 'Business Units'
        }
        
        for category, display_name in category_mapping.items():
            if counts[category]:  # Only process if we found any matches
                unique_matches = len(unique_records[category])
                coverage_percentage = (unique_matches / total_records) * 100
                
                summary_data.append({
                    'Field': col_name,
                    f'{display_name} Found': ', '.join(sorted(counts[category].keys())),
                    'Coverage': f"{unique_matches} ({coverage_percentage:.2f}%)",
                    'Evidence': [evidence[category][c] for c in sorted(counts[category].keys())],
                    'Records Processed': records_processed
                })
        
        return summary_data
    except Exception as e:
        st.error(f"Error processing extractions for field {col_name}: {str(e)}")
        return []

def process_custom_extraction_parallel(col_data, col_name, total_records, category_name, automaton):
    """Process custom extractions for a single column in parallel"""
    try:
        results = []
        unique_matches = set()  # Track unique records with matches
        
        for idx, val in enumerate(col_data):
            val_lower = str(val).lower()
            matches = set()
            for _, (_, match) in automaton.iter(val_lower):
                if is_valid_match(match.lower(), val_lower):
                    matches.add(match)
            if matches:  # If we found any matches
                unique_matches.add(idx)  # Add record index to unique set
            results.append((val, list(matches)))

        match_counts = {}
        match_evidence = {}
        records_processed = len(col_data)

        for val, matches in results:
            for m in matches:
                match_counts[m] = match_counts.get(m, 0) + 1
                if m not in match_evidence:
                    match_evidence[m] = val

        if match_counts:
            unique_match_count = len(unique_matches)
            coverage_percentage = (unique_match_count / total_records) * 100

            return [{
                "Field": col_name,
                f"{category_name}s Found": ', '.join(sorted(match_counts.keys())),
                "Coverage": f"{unique_match_count} ({coverage_percentage:.2f}%)",
                "Evidence": [match_evidence[m] for m in sorted(match_counts.keys())],
                "Records Processed": records_processed
            }]
        return []
    except Exception as e:
        st.error(f"Error processing custom extraction for field {col_name}: {str(e)}")
        return []

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
    
    # Calculate percentages for tooltip
    total = df[x_col].sum()
    df['percentage'] = (df[x_col] / total * 100).round(1)
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f'{x_col}:Q', title=x_col),
        y=alt.Y(f'{y_col}:N', 
                sort='-x', 
                title=y_col,
                axis=alt.Axis(labelLimit=0)),  # This ensures labels are not truncated
        color=alt.Color(f'{x_col}:Q', scale=alt.Scale(scheme=color_scheme)),
        tooltip=[
            alt.Tooltip(f'{y_col}:N', title='Value'),
            alt.Tooltip(f'{x_col}:Q', title='Count'),
            alt.Tooltip('percentage:Q', title='Percentage', format='.1f')
        ]
    ).properties(
        width=600,
        height=max(400, len(df) * 25),  # Dynamic height based on number of bars
        title=title
    )
    
    return chart  # Removed the text layer that was adding labels on bars

def render_donut_chart(df, x_col, y_col, title, color_scheme='blues'):
    """Helper function to create consistent donut charts with proper label handling"""
    # Shorten labels if needed
    df[y_col] = shorten_labels(df[y_col].astype(str))
    
    # Calculate percentages for the labels
    total = df[x_col].sum()
    df['percentage'] = (df[x_col] / total * 100).round(1)
    df['label'] = df[y_col] + ' (' + df[x_col].astype(str) + ', ' + df['percentage'].astype(str) + '%)'
    
    # Define color schemes
    color_schemes = {
        'blues': ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'],
        'greens': ['#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd']
    }
    
    # Get colors based on scheme
    colors = color_schemes.get(color_scheme, color_schemes['blues'])
    # Repeat colors if needed
    colors = colors * (len(df) // len(colors) + 1)
    colors = colors[:len(df)]
    
    # Create the donut chart using Plotly
    fig = go.Figure(data=[go.Pie(
        labels=df['label'],  # Use the combined label
        values=df[x_col],
        hole=.5,
        textinfo='label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>',
        marker=dict(colors=colors)
    )])
    
    fig.update_traces(
        textposition='outside'
    )
    
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=500,
        width=800,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

# Function to save connection details
def save_connection_details(connection_name, details):
    """Save database connection details to a JSON file"""
    try:
        # Create connections directory if it doesn't exist
        os.makedirs('connections', exist_ok=True)
        
        # Load existing connections
        connections = {}
        if os.path.exists('connections/db_connections.json'):
            with open('connections/db_connections.json', 'r') as f:
                connections = json.load(f)
        
        # Add new connection
        connections[connection_name] = details
        
        # Save updated connections
        with open('connections/db_connections.json', 'w') as f:
            json.dump(connections, f, indent=4)
            
        return True
    except Exception as e:
        st.error(f"Error saving connection details: {str(e)}")
        return False

# Function to load connection details
def load_connection_details():
    """Load saved database connection details from JSON file"""
    try:
        if os.path.exists('connections/db_connections.json'):
            with open('connections/db_connections.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading connection details: {str(e)}")
        return {}

# Function to delete connection details
def delete_connection_details(connection_name):
    """Delete a saved database connection"""
    try:
        if os.path.exists('connections/db_connections.json'):
            with open('connections/db_connections.json', 'r') as f:
                connections = json.load(f)
            
            if connection_name in connections:
                del connections[connection_name]
                
                with open('connections/db_connections.json', 'w') as f:
                    json.dump(connections, f, indent=4)
                return True
        return False
    except Exception as e:
        st.error(f"Error deleting connection details: {str(e)}")
        return False

st.set_page_config(page_title="DataSleuth", layout="wide", initial_sidebar_state="expanded", page_icon="static/favicon_io/favicon-32x32.png")

# Dark mode style
dark_style = """
<style>
/* Cool wave animation loader styles */
.spinner-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: var(--background-color);
    z-index: 9999;
    display: flex;
    justify-content: center;
    align-items: center;
}

.loader {
    transform: rotateZ(45deg);
    perspective: 1000px;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    color: #6ae285;  /* Using our green color */
}

.loader:before,
.loader:after {
    content: '';
    display: block;
    position: absolute;
    top: 0;
    left: 0;
    width: inherit;
    height: inherit;
    border-radius: 50%;
    transform: rotateX(70deg);
    animation: 1s spin linear infinite;
}

.loader:after {
    color: #54b6ed;  /* Using our blue color */
    transform: rotateY(70deg);
    animation-delay: .4s;
}

@keyframes rotate {
    0% {
        transform: translate(-50%, -50%) rotateZ(0deg);
    }
    100% {
        transform: translate(-50%, -50%) rotateZ(360deg);
    }
}

@keyframes rotateccw {
    0% {
        transform: translate(-50%, -50%) rotate(0deg);
    }
    100% {
        transform: translate(-50%, -50%) rotate(-360deg);
    }
}

@keyframes spin {
    0%,
    100% {
        box-shadow: .2em 0px 0 0px currentcolor;
    }
    12% {
        box-shadow: .2em .2em 0 0 currentcolor;
    }
    25% {
        box-shadow: 0 .2em 0 0px currentcolor;
    }
    37% {
        box-shadow: -.2em .2em 0 0 currentcolor;
    }
    50% {
        box-shadow: -.2em 0 0 0 currentcolor;
    }
    62% {
        box-shadow: -.2em -.2em 0 0 currentcolor;
    }
    75% {
        box-shadow: 0px -.2em 0 0 currentcolor;
    }
    87% {
        box-shadow: .2em -.2em 0 0 currentcolor;
    }
}

/* Theme-aware variables */
:root {
    --background-color: var(--background-color, #ffffff);
    --secondary-background-color: var(--secondary-background-color, #f0f2f6);
    --text-color: var(--text-color, #262730);
    --border-color: var(--border-color, #e6e6e6);
    --accent-color: var(--accent-color, #00acb5);
}

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
    :root {
        --background-color: #121212;
        --secondary-background-color: #1a1a1a;
        --text-color: #e0e0e0;
        --border-color: #333;
        --accent-color: #00ff00;
    }
}
</style>
"""
st.markdown(dark_style, unsafe_allow_html=True)

# Add analysis time tracking
if 'analysis_start_time' not in st.session_state:
    st.session_state.analysis_start_time = time.time()

# Initialize loading screen state
if 'loading_complete' not in st.session_state:
    st.session_state.loading_complete = False

# Show loading screen if not complete
if not st.session_state.loading_complete:
    # Create spinner container
    spinner_html = """
    <div class="spinner-container">
        <div class="loader"></div>
    </div>
    """
    
    st.markdown(spinner_html, unsafe_allow_html=True)
    
    # Wait for a short duration
    time.sleep(1.5)
    
    # Mark loading as complete
    st.session_state.loading_complete = True
    st.rerun()
#st.title("📊 DataSleuth - Smart EDA Viewer")

st.info("ℹ️ To load new data, please refresh the page first to clear the current session.")
st.markdown('<div id="load-previous"></div>', unsafe_allow_html=True)
st.markdown("## Load Previous Session")
uploaded_session = st.file_uploader("📂 Load Previous Session", type=["pkl"])
if uploaded_session:
    # Check if we've already loaded this session
    if 'session_loaded' not in st.session_state:
        session_data = pickle.load(uploaded_session)
        df = session_data["dataframe"]
        primary_keys = session_data["primary_keys"]
        countries_input = session_data["countries_input"]
        regions_input = session_data["regions_input"]

        # Store the dataframe in session state
        st.session_state.df = df
        st.session_state.file_name = "loaded_session.pkl"  # Set a default name for loaded sessions
        st.session_state.session_loaded = True  # Mark session as loaded

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
        st.rerun()  # Force a rerun to trigger analytics

# Add Database Loading Section
st.markdown('<div id="load-database"></div>', unsafe_allow_html=True)
st.markdown("## Load from Database")
with st.expander("📊 Database Connection Options", expanded=False):
    db_type = st.selectbox(
        "Select Database Type",
        ["Spark Thrift Server"],
        key="db_type"
    )

    if db_type == "Spark Thrift Server":
        st.markdown("### Spark Thrift Server Connection")
        
        # Load saved connections
        saved_connections = load_connection_details()
        
        # Connection management section
        st.markdown("#### 🔧 Connection Management")
        
        # Show saved connections
        if saved_connections:
            st.markdown("##### Saved Connections")
            for conn_name in saved_connections:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(conn_name)
                with col2:
                    if st.button("Delete", key=f"del_{conn_name}"):
                        if delete_connection_details(conn_name):
                            st.success(f"Deleted connection: {conn_name}")
                            st.rerun()
        
        # Add new connection form
        st.markdown("##### Save New Connection")
        with st.form(key="save_connection_form"):
            new_conn_name = st.text_input("Connection Name")
            new_host = st.text_input("Host")
            new_port = st.number_input("Port", value=10000, min_value=1, max_value=65535)
            new_use_auth = st.checkbox("Use Authentication")
            
            # Create columns for username/password to keep them side by side
            auth_col1, auth_col2 = st.columns(2)
            with auth_col1:
                new_username = st.text_input("Username", value="", key="new_username")
            with auth_col2:
                new_password = st.text_input("Password", type="password", value="", key="new_password")
            
            if st.form_submit_button("Save Connection"):
                if new_conn_name and new_host and new_port:
                    details = {
                        "host": new_host,
                        "port": new_port,
                        "use_auth": new_use_auth,
                        "username": new_username if new_use_auth else "",
                        "password": new_password if new_use_auth else ""
                    }
                    if save_connection_details(new_conn_name, details):
                        st.success(f"Saved connection: {new_conn_name}")
                        st.rerun()
                else:
                    st.error("Please fill in all required fields")
        
        st.markdown("---")
        
        # Connection selection and query form
        if saved_connections:
            selected_conn = st.selectbox(
                "Select Connection",
                options=list(saved_connections.keys())
            )
            
            # Use saved connection details
            conn_details = saved_connections[selected_conn]
            
            # Show connection details
            st.info(f"Using connection: {selected_conn}")
            
            st.markdown(f"""
            - Host: {conn_details['host']}
            - Port: {conn_details['port']}
            - Authentication: {'Enabled' if conn_details['use_auth'] else 'Disabled'}
            """)
            
            # Query form
            with st.form(key="saved_connection_form"):
                query = st.text_area("SQL Query", value="SELECT * FROM database.table", 
                                   help="Enter your query in format: SELECT * FROM database.table")
                
                if st.form_submit_button("Connect and Load Data"):
                    try:
                        # Import required packages
                        import pandas as pd
                        
                        # Create connection with or without authentication
                        conn_params = {
                            'host': conn_details['host'],
                            'port': conn_details['port']
                        }
                        
                        if conn_details['use_auth']:
                            conn_params.update({
                                'username': conn_details['username'],
                                'password': conn_details['password'],
                                'auth': 'LDAP'
                            })
                        else:
                            conn_params['auth'] = 'NONE'
                        
                        # Show loading message
                        loading_msg = st.info("Loading data from database... This may take a few moments.")
                        
                        # Single database call to fetch all data
                        with hive.Connection(**conn_params) as conn:
                            df = pd.read_sql(query, conn)
                        
                        # Clear loading message
                        loading_msg.empty()
                        
                        # Store the dataframe in session state
                        st.session_state.df = df
                        st.session_state.file_name = "spark_query_result.csv"
                        
                        st.success(f"✅ Successfully loaded {df.shape[0]:,} records with {df.shape[1]} fields from Spark Thrift Server.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error connecting to Spark Thrift Server: {str(e)}")
        else:
            st.warning("⚠️ No saved connections found. Please save a connection first.")

# Add logo to sidebar
st.sidebar.image("static/logo.png", use_container_width=False, width=250)

def create_toc():
    """Create a dynamic table of contents based on application state"""
    toc_items = []
    
    # Always show these items
    toc_items.append(("📂 Load Previous Session", "load-previous"))
    toc_items.append(("💾 Load from Database", "load-database"))
    toc_items.append(("📤 Upload New File", "upload-file"))
    
    # Show additional items if data is loaded from any source
    if ('uploaded_file' in st.session_state and st.session_state.uploaded_file is not None) or \
       ('df' in locals() and df is not None) or \
       ('df' in st.session_state and st.session_state.df is not None):
        toc_items.append(("📊 Field-wise Summary", "field-summary"))
        toc_items.append(("🔑 Primary Key Identification", "primary-key"))
        toc_items.append(("🔍 Per Field Insights", "field-insights"))
        toc_items.append(("🔎 Pattern Detection", "pattern-detection"))
        toc_items.append(("📈 Outlier Detection", "outlier-detection"))
        toc_items.append(("📊 Advanced Outlier Detection", "advanced-outlier"))
        
        # Only show built-in extraction if enabled
        if 'sidebar_visible' in st.session_state and st.session_state.sidebar_visible:
            toc_items.append(("🌍 Built-in Extraction Insights", "builtin-extraction"))
        
        # Only show custom extraction if categories exist
        if "custom_categories" in st.session_state and st.session_state.custom_categories:
            toc_items.append(("🔧 Custom Extraction Insights", "custom-extraction-insights"))
        
        # Add Save Session at the end when data is loaded
        toc_items.append(("💾 Save Session", "save-session"))
    
    return toc_items

# Create TOC header with refresh button
col1, col2 = st.sidebar.columns([6, 1])
with col1:
    st.markdown("### 📑 Table of Contents")
with col2:
    if st.button("↻", help="Update Table of Contents", key="refresh_toc"):
        st.rerun()

toc_items = create_toc()

for label, anchor in toc_items:
    st.sidebar.markdown(f'<a href="#{anchor}">{label}</a>', unsafe_allow_html=True)

st.sidebar.markdown("---")  # Add a separator

# Initialize session state for inputs if not exists
if 'countries_input' not in st.session_state:
    st.session_state.countries_input = "India, Bharat, Republic of India, United Arab Emirates, UAE, Emirates, Saudi Arabia, KSA, Kingdom of Saudi Arabia, United Kingdom, UK, Britain, Great Britain, United States of America, USA, US, United States, America, Armenia, Republic of Armenia, Azerbaijan, Republic of Azerbaijan, Canada, Côte d'Ivoire, Ivory Coast, Chile, Republic of Chile, Colombia, Republic of Colombia, Costa Rica, Republic of Costa Rica, Germany, Deutschland, Federal Republic of Germany, Ecuador, Republic of Ecuador, Egypt, Arab Republic of Egypt, Spain, España, Kingdom of Spain, France, French Republic, Georgia, Sakartvelo, Ghana, Republic of Ghana, Croatia, Republic of Croatia, Italy, Italian Republic, Japan, Nippon, Nihon, Republic of Korea, South Korea, Korea (South), Lithuania, Republic of Lithuania, Luxembourg, Grand Duchy of Luxembourg, Morocco, Kingdom of Morocco, TFYR Macedonia, North Macedonia, Macedonia, Mexico, United Mexican States, Netherlands, Holland, Kingdom of the Netherlands, Philippines, Republic of the Philippines, Peru, Republic of Peru, Poland, Republic of Poland, Portugal, Portuguese Republic, Romania, Senegal, Republic of Senegal, Suriname, Republic of Suriname, Togo, Togolese Republic, Thailand, Kingdom of Thailand, Siam, Turkey, Türkiye, Republic of Turkey, Ethiopia, Federal Democratic Republic of Ethiopia, Algeria, People's Democratic Republic of Algeria, Jordan, Hashemite Kingdom of Jordan, Madagascar, Republic of Madagascar, Kazakhstan, Republic of Kazakhstan, China, People's Republic of China, PRC, Lebanon, Lebanese Republic, Serbia, Republic of Serbia, South Africa, Republic of South Africa, United Republic of Tanzania, Tanzania, Cameroon, Republic of Cameroon, Russian Federation, Russia, Switzerland, Swiss Confederation, Viet Nam, Vietnam, Socialist Republic of Vietnam, Nigeria, Federal Republic of Nigeria, Indonesia, Republic of Indonesia, Uganda, Republic of Uganda, Ukraine, Rwanda, Republic of Rwanda, Gabon, Gabonese Republic, Belarus, Kenya, Republic of Kenya, Kosovo, Republic of Kosovo, Tunisia, Republic of Tunisia, Uzbekistan, Republic of Uzbekistan, Albania, Republic of Albania, Jamaica, CTSS, Argentina, Argentine Republic, Australia, Commonwealth of Australia, Bosnia and Herzegovina, BiH, Belgium, Kingdom of Belgium, Brazil, Federative Republic of Brazil, Czech Republic, Czechia, Denmark, Kingdom of Denmark, Dominican Republic, Finland, Republic of Finland, Greece, Hellenic Republic, Mauritius, Republic of Mauritius, Guatemala, Republic of Guatemala, Guyana, Co-operative Republic of Guyana, Honduras, Republic of Honduras, Ireland, Éire, Republic of Ireland, Malaysia, Nicaragua, Republic of Nicaragua, Norway, Kingdom of Norway, Sweden, Kingdom of Sweden, Singapore, Republic of Singapore, El Salvador, Republic of El Salvador, Estonia, Republic of Estonia"
if 'regions_input' not in st.session_state:
    st.session_state.regions_input = "APAC, EMEA, EWAP, Global, INDIA, LATAM, MAJOREL, Specialized Services, TGI"
if 'compliance_input' not in st.session_state:
    st.session_state.compliance_input = "GDPR, CCPA, HIPAA, PCI, PCI DSS, ISO 27001, SOC 2, NIST, FISMA, GLBA, SOX, FedRAMP, CMMC, NIST 800-53, NIST 800-171, ISO 27701, ISO 22301, ISO 31000, ISO 9001, ISO 14001, ISO 45001, ISO 20000, ISO 27017, ISO 27018, ISO 27002, ISO 27005, ISO 27006, ISO 27007, ISO 27008, ISO 27009, ISO 27010, ISO 27011, ISO 27012, ISO 27013, ISO 27014, ISO 27015, ISO 27016, ISO 27019, ISO 27020, ISO 27021, ISO 27022, ISO 27023, ISO 27024, ISO 27025, ISO 27026, ISO 27027, ISO 27028, ISO 27029, ISO 27030, ISO 27031, ISO 27032, ISO 27033, ISO 27034, ISO 27035, ISO 27036, ISO 27037, ISO 27038, ISO 27039, ISO 27040, ISO 27041, ISO 27042, ISO 27043, ISO 27044, ISO 27045, ISO 27046, ISO 27047, ISO 27048, ISO 27049, ISO 27050, ISO 27051, ISO 27052, ISO 27053, ISO 27054, ISO 27055, ISO 27056, ISO 27057, ISO 27058, ISO 27059, ISO 27060, ISO 27061, ISO 27062, ISO 27063, ISO 27064, ISO 27065, ISO 27066, ISO 27067, ISO 27068, ISO 27069, ISO 27070, ISO 27071, ISO 27072, ISO 27073, ISO 27074, ISO 27075, ISO 27076, ISO 27077, ISO 27078, ISO 27079, ISO 27080, ISO 27081, ISO 27082, ISO 27083, ISO 27084, ISO 27085, ISO 27086, ISO 27087, ISO 27088, ISO 27089, ISO 27090, ISO 27091, ISO 27092, ISO 27093, ISO 27094, ISO 27095, ISO 27096, ISO 27097, ISO 27098, ISO 27099, ISO 27100"
if 'business_unit_input' not in st.session_state:
    st.session_state.business_unit_input = "IT-SOFTWARE, CLIENT OPERATIONS, WORKFORCE, STAFF, CS-CLIENT SERVICES, BUSINESS DEVELOPMENT, OPS-CLIENT DELIVERY, HR, Client Operations, Workforce Management, WORKFORCE MANAGEMENT, Facilities, SALES, LC-AUDIT, FA-FP&A, HR-RECRUITMENT / TALENT ACQUISITION, Ops-Client Delivery, HR, MANAGEMENT/MANAGERS, Client Services, OPS-WORKFORCE MANAGEMENT, LEGAL, AF-PREMISES AND ADMINISTRATION, SUPPORT HELP DESK, HR-CROSS FUNCTION ROLES, Infrastructure Desktop, Quality Assurance, OPS-GLOBAL PROCESSES, STANDARDS AND CONTINUOUS IMPROVEMENT, SUPPORT ANALYST, DEVELOPMENT DEVELOPER, IT-INFRASTRUCTURE OPERATIONS, IT-INFORMATION SECURITY, Training, MANAGEMENT/MANAGERS, CLIENT, SUPPORT ADMINISTRATION, HR-TRAINING, IT, HUMAN RESOURCES, BUSINESS INTELLIGENCE, HR-PAYROLL, HR-LEARNING AND DEVELOPMENT AND ORGANIZATIONAL DEVELOPMENT, INFRASTRUCTURE DESKTOP, OPS-BUSINESS INTELLIGENCE AND REPORTING, TRAINING, RECRUITING, OPS-QUALITY ANALYSIS / CONTINUOUS IMPROVEMENT, STAFF, STAFF, OPS-CROSS FUNCTION ROLES, IT-SERVICE DESK AND IT SERVICE MANAGEMENT, QUALITY ASSURANCE, FINANCE, IT-SUPPORT SERVICES, FA-PROCUREMENT AND SUPPORT, MKT- CROSS FUNCTION ROLES, IT-DATA COE, MANAGEMENT/MANAGERS, TRAINING, INFRASTRUCTURE SYSTEMS, Support Help Desk, MARKETING, HR-ONBOARDING, AF-HEALTH AND SAFETY, Professional Services, FACILITIES, AF-MAINTENANCE, ADMINISTRATION, DEVELOPMENT SCRIPTING, BD-BUSINESS DEVELOPMENT, HR-Training, CLIENT SERVICES, CS-STRATEGIC ACCOUNT MANAGEMENT, IT-INFRASTRUCTURE ARCHITECTURE AND ENGINEERING, MANAGEMENT/MANAGERS, HR, SECURITY, PAYROLL, MANAGEMENT TRAINING DEVELOPMENT, ANALYST, DS-CROSS FUNCTION ROLES, RISK, Ops-Quality Analysis / Continuous Improvement, Finance, OPS-PROJECT MANAGEMENT, OPS-Cross Function Roles, PROCUREMENT, INFORMATION SECURITY, HR-EMPLOYEE RELATIONS, EXECUTIVE MANAGEMENT, FA-CROSS FUNCTION ROLES, LC-COMPLIANCE, IT-Service Desk and IT Service Management, BD-CROSS FUNCTION ROLES, APPLICATION SUPPORT, Ops-Workforce Management, STDS-ROLLOUT & AUDIT, DS-CONSULTING AND SOLUTIONING, IT-Infrastructure Operations, CS-Client Services, IT-Software, HR-HELP DESK, INFRASTRUCTURE TELECOM, HR-COMPENSATION AND BENEFITS, MKT-DIGITAL MARKETING, INFRASTRUCTURE NETWORK, LC-LEGAL, Business Development, HR-Cross Function Roles, FA-Cross Function Roles, FA-FINANCIAL SYSTEMS, TRANSFORMATION AND AUTOMATION, BD-SALES ENABLEMENT, MKT-WEB DESIGN, HR-Payroll, Marketing, IT-CROSS FUNCTION ROLES, BD-PRESALES, INFRASTRUCTURE DATABASE, Recruiting, BD-Business Development, SALES, CLIENT, EM-EXECUTIVE ASSISTANTS, IT, STAFF, Development Scripting, TRANSFORMATION, EXTERNAL, IT, MANAGEMENT/MANAGERS, HR-DIVERSITY, EQUITY AND INCLUSION, Management Training Development, CORPORATE COMPLIANCE, Infrastructure Database, Human Resources, SUPPORT PRODUCT MANAGEMENT, AF-CROSS FUNCTION ROLES, DS-DATA ANALYTICS, Transformation, SUPPORT PROJECT MANAGEMENT, Risk, STDS-IMPROVEMENT DELIVERY PERFORMANCE, STAFF, TRAINING, IT-Infrastructure Architecture and Engineering, Infrastructure Systems, STRATEGIC ACCOUNT MANAGEMENT, IT-PLATFORM ENGINEERING SOLUTIONS COE, BUSINESS OPERATIONS, AF-LOGISTICS AND WAREHOUSING, DS-Consulting and Solutioning, MKT-CONTENT DEVELOPMENT, EM-LOCAL EXECUTIVE MANAGEMENT TIER 3, STAFF, STAFF, STAFF, Ops-Business Intelligence and Reporting, Legal, EM-LOCAL EXECUTIVE MANAGEMENT, FA-TREASURY, STAFF, IT, Support Product Management, LC-DATA PRIVACY, AUDITOR, Business Operations, Ops-Project Management, Support Project Management, HR, MANAGEMENT/MANAGERS, TRAINING, Business Intelligence, EM-LOCAL EXECUTIVE MANAGEMENT TIER 1, DEVELOPMENT ARCHITECT, AF-Premises and Administration, TRAINING, MANAGEMENT/MANAGERS, Payroll, Procurement, AF-Maintenance, Ops-Global Processes, Standards and Continuous Improvement, IT, STAFF, STAFF, HR-Onboarding, IT, TRAINING, HR-Learning and Development and Organizational Development"

st.sidebar.markdown("### ⚙️ Extraction Configs")

with st.sidebar.expander("Built-in Extraction Configs", expanded=False):
    sidebar_visible = st.checkbox("Enable Built-in Extraction Configs", value=True, key='sidebar_visible')

    if sidebar_visible:
        countries_input = st.text_area(
            "Country List (comma separated)",
            value=st.session_state.countries_input
        )
        st.session_state.countries_input = countries_input
        
        regions_input = st.text_area(
            "Region List (comma separated)",
            value=st.session_state.regions_input
        )
        st.session_state.regions_input = regions_input

        compliance_input = st.text_area(
            "Compliance List (comma separated)",
            value=st.session_state.compliance_input
        )
        st.session_state.compliance_input = compliance_input

        business_unit_input = st.text_area(
            "Business Unit List (comma separated)",
            value=st.session_state.business_unit_input
        )
        st.session_state.business_unit_input = business_unit_input

        COUNTRY_LIST = [x.strip() for x in st.session_state.countries_input.split(",")]
        REGION_LIST = [x.strip() for x in st.session_state.regions_input.split(",")]
        COMPLIANCE_LIST = [x.strip() for x in st.session_state.compliance_input.split(",")]
        BUSINESS_UNIT_LIST = [x.strip() for x in st.session_state.business_unit_input.split(",")]

        # Build automatons once
        COUNTRY_AUTOMATON = build_automaton(COUNTRY_LIST)
        REGION_AUTOMATON = build_automaton(REGION_LIST)
        COMPLIANCE_AUTOMATON = build_automaton(COMPLIANCE_LIST)
        BUSINESS_UNIT_AUTOMATON = build_automaton(BUSINESS_UNIT_LIST)

# Add Custom Extraction Category section
with st.sidebar.expander("➕ Add Custom Extraction Categories", expanded=False):
    # Initialize form state if not exists
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False

    with st.form(key="custom_extraction_form"):
        # Only show empty inputs if form was just submitted
        if st.session_state.form_submitted:
            custom_category = st.text_input("Category Name (e.g., Product, Company)", value="", key="category_input")
            custom_keywords_input = st.text_area("Keywords (comma separated)", value="", key="keywords_input")
            st.session_state.form_submitted = False
        else:
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
                    # Set flag to clear form on next render
                    st.session_state.form_submitted = True
                    # Force a rerun to update the TOC
                    st.rerun()
            else:
                st.error("❌ Please enter both a category name and at least one keyword.")

# Display current custom categories outside the main expander
if "custom_categories" in st.session_state and st.session_state.custom_categories:
    st.sidebar.markdown("### 🗂️ Current Custom Categories")
    # Create a list of categories to iterate over
    categories = list(st.session_state.custom_categories.items())
    for cat_name, meta in categories:
        with st.sidebar.expander(f"🔧 `{cat_name}` ({len(meta['keywords'])} keywords)"):
            st.write(", ".join(meta["keywords"]))

            # Optional: edit or delete
            new_keywords = st.text_area(f"✏️ Edit keywords for `{cat_name}`", value=", ".join(meta["keywords"]), key=f"edit_{cat_name}")
            if st.button(f"Update `{cat_name}`", key=f"update_{cat_name}"):
                new_kw_list = [kw.strip() for kw in new_keywords.split(",") if kw.strip()]
                st.session_state.custom_categories[cat_name]["keywords"] = new_kw_list
                st.session_state.custom_categories[cat_name]["automaton"] = build_automaton(new_kw_list)
                st.success(f"✅ Updated keywords for `{cat_name}`")
                # Force a rerun to update the TOC
                st.rerun()

            if st.button(f"❌ Delete `{cat_name}`", key=f"delete_{cat_name}"):
                del st.session_state.custom_categories[cat_name]
                st.warning(f"🗑️ Category `{cat_name}` has been deleted from the system. The UI will update on your next interaction.")
                # Force a rerun to update the TOC
                st.rerun()

st.sidebar.markdown("---")  # Add a separator

# Initialize session state for filters if not exists
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {}
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

# File Upload Section
st.markdown('<div id="upload-file"></div>', unsafe_allow_html=True)
st.markdown("## Upload New File")
uploaded_file = st.file_uploader("Upload a CSV, Excel, JSON, or XML file", type=["csv", "xlsx", "json", "xml"])

if uploaded_file is not None:
    # Store the uploaded file in session state
    st.session_state.uploaded_file = uploaded_file
    st.session_state.file_name = uploaded_file.name

# Check if we have a stored file in session state
if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file
    
    # Reset the file pointer to the beginning
    uploaded_file.seek(0)
    
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        try:
            # Try to read as JSON first
            json_data = json.load(uploaded_file)
            
            # Function to flatten nested JSON structure
            def flatten_json(data, parent_key='', sep='_'):
                items = []
                if isinstance(data, dict):
                    for k, v in data.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, (dict, list)):
                            items.extend(flatten_json(v, new_key, sep=sep).items())
                        else:
                            items.append((new_key, v))
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                        if isinstance(item, (dict, list)):
                            items.extend(flatten_json(item, new_key, sep=sep).items())
                        else:
                            items.append((new_key, item))
                return dict(items)
            
            # If the JSON is a list of objects, process each object
            if isinstance(json_data, list):
                flattened_data = [flatten_json(item) for item in json_data]
                df = pd.DataFrame(flattened_data)
            else:
                # If it's a single object, flatten it and create a single-row DataFrame
                flattened_data = flatten_json(json_data)
                df = pd.DataFrame([flattened_data])
                
        except Exception as e:
            # If that fails, try to read as JSON lines
            try:
                # Reset file pointer again before trying JSON lines
                uploaded_file.seek(0)
                df = pd.read_json(uploaded_file, lines=True)
            except Exception as e2:
                st.error(f"Error reading JSON file: {str(e2)}")
                st.info("Please ensure your JSON file is either a valid JSON array of objects or JSON Lines format.")
                df = None
    elif uploaded_file.name.endswith(".xml"):
        try:
            # Read the XML file content
            xml_content = uploaded_file.read()
            
            # Reset file pointer for future reads
            uploaded_file.seek(0)
            
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
    
    # Store the dataframe in session state
    st.session_state.df = df
    st.success(f"✅ Loaded **{df.shape[0]}** records with **{df.shape[1]}** fields.")
elif "df" in st.session_state:
    df = st.session_state.df
    st.info("ℹ️ Using data loaded from session file.")
else:
    df = None
    st.info("📂 Please upload a file to begin analysis.")

# Create a form for filter selections
if df is not None:
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
        
        # Handle form submission inside the form
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
                # Reset processed fields to force reprocessing with filtered data
                st.session_state.processed_fields = {}
            else:
                st.session_state.filtered_data = None
                st.session_state.active_filters = {}
                # Reset processed fields to force reprocessing with original data
                st.session_state.processed_fields = {}
            st.rerun()
        
        elif clear_filters:
            st.session_state.active_filters = {}
            st.session_state.filtered_data = None
            # Reset processed fields to force reprocessing with original data
            st.session_state.processed_fields = {}
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

    st.markdown('<div id="field-summary"></div>', unsafe_allow_html=True)
    st.markdown("## Field-wise Summary")
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

    st.markdown('<div id="primary-key"></div>', unsafe_allow_html=True)
    st.markdown("## Primary Key Identification")

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

    st.markdown('<div id="field-insights"></div>', unsafe_allow_html=True)
    st.markdown("## Per Field Insights")
    
    # Get the number of CPU cores available
    num_cores = multiprocessing.cpu_count()
    # Use 75% of available cores to avoid overwhelming the system
    num_workers = max(1, int(num_cores * 0.75))
    
    # Initialize session state for batch processing if not exists
    if 'processed_fields' not in st.session_state:
        st.session_state.processed_fields = {}
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 5
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = 0
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "View All Fields (Paginated)"
    if 'selected_field' not in st.session_state:
        st.session_state.selected_field = None
    
    # Get all fields
    all_fields = list(df.columns)
    total_fields = len(all_fields)

    # Add field selection option
    st.markdown("### 🔍 Field Selection")
    view_mode = st.radio(
        "Choose how to view field insights:",
        ["View All Fields (Paginated)", "Select Specific Field"],
        horizontal=True,
        key="view_mode_radio",
        index=0 if st.session_state.view_mode == "View All Fields (Paginated)" else 1
    )
    
    # Update session state view mode
    st.session_state.view_mode = view_mode

    if view_mode == "Select Specific Field":
        # If we have a previously selected field and it still exists in the filtered data, use it
        if st.session_state.selected_field and st.session_state.selected_field in all_fields:
            default_field = st.session_state.selected_field
        else:
            default_field = all_fields[0] if all_fields else None
            st.session_state.selected_field = default_field

        selected_field = st.selectbox(
            "Select a field to analyze:",
            options=all_fields,
            key="field_selector",
            index=all_fields.index(default_field) if default_field else 0
        )
        
        # Update session state selected field
        st.session_state.selected_field = selected_field
        
        # Process the selected field if not already processed
        if selected_field not in st.session_state.processed_fields:
            st.info(f"Processing field: {selected_field}...")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                process_func = partial(process_single_column, 
                                     total_records=len(df),
                                     primary_keys=primary_keys if 'primary_keys' in locals() else None,
                                     original_df=df)
                future = executor.submit(process_func, df[selected_field], selected_field)
                try:
                    insights = future.result()
                    st.session_state.processed_fields[selected_field] = insights
                except Exception as e:
                    st.error(f"Error processing column {selected_field}: {str(e)}")
        
        # Display insights for selected field
        if selected_field in st.session_state.processed_fields:
            insights = st.session_state.processed_fields[selected_field]
            
            # Display the insights for this column
            st.markdown(f"### 🧬 {insights['column_name']}")
            
            if 'error' in insights:
                st.error(f"Error processing column: {insights['error']}")
            else:
                # Display coverage
                if insights['text_content']:
                    st.progress(float(insights['text_content'][0].split(': ')[1].strip('%')) / 100,
                              text=insights['text_content'][0])
                
                # Handle datetime columns
                if insights.get('is_datetime'):
                    st.markdown("#### 📈 Trend (Date/Time Field)")
                    parsed_col = insights['parsed_datetime']
                    
                    # Create datetime visualization UI
                    freq = st.selectbox(
                        f"Choose trend resolution for `{selected_field}`",
                        options=["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                        index=0,
                        key=f"freq_{selected_field}"
                    )
                    
                    freq_map = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "M",
                        "Quarterly": "Q",
                        "Yearly": "Y"
                    }
                    
                    show_all_dates = st.checkbox(
                        f"Show all dates for `{selected_field}`",
                        value=False,
                        key=f"show_all_{selected_field}"
                    )
                    
                    # Get the original index of non-null datetime values
                    valid_datetime_mask = parsed_col.notna()
                    original_indices = parsed_col.index[valid_datetime_mask]
                    
                    # Create a DataFrame with the datetime column and original index
                    datetime_df = pd.DataFrame({
                        '__datetime__': parsed_col[valid_datetime_mask]
                    }, index=original_indices)
                    
                    if not show_all_dates:
                        min_date = parsed_col.min().date()
                        max_date = parsed_col.max().date()
                        start_date, end_date = st.date_input(
                            f"Select date range for `{selected_field}`",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key=f"range_{selected_field}"
                        )
                        
                        # Filter the datetime DataFrame using boolean indexing
                        date_mask = (datetime_df['__datetime__'].dt.date >= start_date) & \
                                  (datetime_df['__datetime__'].dt.date <= end_date)
                        filtered_df = datetime_df[date_mask].copy()
                    else:
                        filtered_df = datetime_df.copy()
                    
                    # Resample and count records
                    resampled = filtered_df.set_index('__datetime__').resample(freq_map[freq])
                    record_counts = resampled.size().rename("Total Records")
                    chart_df = pd.DataFrame(record_counts)
                    
                    if primary_keys:
                        try:
                            # Create a DataFrame with primary keys and datetime
                            pk_datetime_df = pd.DataFrame({
                                '__datetime__': filtered_df['__datetime__']
                            })
                            
                            # Add primary keys by using the current dataframe
                            for pk in primary_keys:
                                if pk in df.columns:
                                    # Get the values using the filtered indices
                                    if isinstance(filtered_df.index, pd.MultiIndex):
                                        # For multi-level index, get the first level
                                        idx = filtered_df.index.get_level_values(0)
                                    else:
                                        idx = filtered_df.index
                                    
                                    # Use loc to get values by index
                                    pk_values = df.loc[idx, pk].values
                                    pk_datetime_df[pk] = pk_values
                            
                            # Drop duplicates based on primary keys and resample
                            unique_keys_df = pk_datetime_df.drop_duplicates(subset=primary_keys)
                            unique_counts = unique_keys_df.set_index('__datetime__').resample(freq_map[freq]).size().rename("Unique Primary Keys")
                            chart_df = chart_df.join(unique_counts, how='outer').fillna(0)
                        except Exception as e:
                            st.warning(f"Could not calculate unique primary keys: {str(e)}")
                    
                    st.line_chart(chart_df)
                
                # Display wordcloud if available
                if 'wordcloud_text' in insights:
                    st.markdown("#### ☁️ Word Cloud (Long Text Field)")
                    try:
                        wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA').generate(insights['wordcloud_text'])
                        fig, ax = plt.subplots(figsize=(10, 5))
                        fig.patch.set_alpha(0)
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"⚠️ Word cloud generation failed: {str(e)}")
                
                # Display charts
                for chart_type, chart_df in insights['charts']:
                    if chart_type == 'bar_chart':
                        chart = create_bar_chart(chart_df, 'Occurrences', selected_field, "Top 10 Values (All Records)")
                        st.altair_chart(chart, use_container_width=True)
                    elif chart_type == 'bar_chart_pk':
                        chart = create_bar_chart(chart_df, 'Occurrences', selected_field, "Top 10 Values (Per Unique Primary Key)", color_scheme='greens')
                        st.markdown("#### Top Values (Per Primary Key)")
                        st.altair_chart(chart, use_container_width=True)
                    elif chart_type == 'donut_chart':
                        fig = render_donut_chart(chart_df, 'Occurrences', selected_field, "Value Distribution (All Records)")
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == 'donut_chart_pk':
                        fig = render_donut_chart(chart_df, 'Occurrences', selected_field, "Value Distribution (Per Unique Primary Key)", color_scheme='greens')
                        st.markdown("#### Value Distribution (Per Primary Key)")
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == 'histogram':
                        hist = alt.Chart(chart_df).mark_bar(color='teal').encode(
                            alt.X(f"{selected_field}:Q", bin=alt.Bin(maxbins=30), title=selected_field),
                            y=alt.Y('count()', title='Count')
                        ).properties(
                            width=600,
                            height=300,
                            title="Distribution"
                        )
                        st.altair_chart(hist, use_container_width=True)
                
                # Display tables
                for table_type, table_df in insights['tables']:
                    if table_type == 'value_counts':
                        st.markdown("### 📊 Value Counts and Percentages")
                        st.dataframe(table_df, use_container_width=True)
                    elif table_type == 'value_counts_pk':
                        st.markdown("### 📊 Value Counts and Percentages (Per Primary Key)")
                        st.dataframe(table_df, use_container_width=True)
                    elif table_type == 'Unique Values':
                        with st.expander("📋 View Unique Values (with counts)"):
                            st.dataframe(table_df, use_container_width=True)
    else:
        # Calculate start and end indices for current batch
        start_idx = st.session_state.current_batch * st.session_state.batch_size
        end_idx = min(start_idx + st.session_state.batch_size, total_fields)
        
        # Display current batch
        if start_idx < total_fields:
            current_batch = all_fields[start_idx:end_idx]
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process only the current batch of fields if not already processed
            fields_to_process = [col for col in current_batch if col not in st.session_state.processed_fields]
            if fields_to_process:
                st.info(f"Processing {len(fields_to_process)} fields in current batch...")
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Create a partial function with the common arguments
                    process_func = partial(process_single_column, 
                                         total_records=len(df),
                                         primary_keys=primary_keys if 'primary_keys' in locals() else None,
                                         original_df=df)
                    
                    # Submit only the fields that need processing
                    future_to_col = {
                        executor.submit(process_func, df[col], col): col 
                        for col in fields_to_process
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_col):
                        col = future_to_col[future]
                        try:
                            insights = future.result()
                            st.session_state.processed_fields[col] = insights
                        except Exception as e:
                            st.error(f"Error processing column {col}: {str(e)}")
            
            # Display insights for current batch
            for i, col in enumerate(current_batch):
                # Calculate overall progress including previous batches
                overall_progress = (start_idx + i + 1) / total_fields
                progress_bar.progress(overall_progress)
                status_text.text(f"Displaying column {start_idx + i + 1}/{total_fields}: {col}")
                
                if col in st.session_state.processed_fields:
                    insights = st.session_state.processed_fields[col]
                    
                    # Display the insights for this column
                    st.markdown(f"### 🧬 {insights['column_name']}")
                    
                    if 'error' in insights:
                        st.error(f"Error processing column: {insights['error']}")
                        continue
                    
                    # Display coverage
                    if insights['text_content']:
                        st.progress(float(insights['text_content'][0].split(': ')[1].strip('%')) / 100,
                                  text=insights['text_content'][0])
                    
                    # Handle datetime columns
                    if insights.get('is_datetime'):
                        st.markdown("#### 📈 Trend (Date/Time Field)")
                        parsed_col = insights['parsed_datetime']
                        
                        # Create datetime visualization UI
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
                        
                        show_all_dates = st.checkbox(
                            f"Show all dates for `{col}`",
                            value=False,
                            key=f"show_all_{col}"
                        )
                        
                        # Get the original index of non-null datetime values
                        valid_datetime_mask = parsed_col.notna()
                        original_indices = parsed_col.index[valid_datetime_mask]
                        
                        # Create a DataFrame with the datetime column and original index
                        datetime_df = pd.DataFrame({
                            '__datetime__': parsed_col[valid_datetime_mask]
                        }, index=original_indices)
                        
                        if not show_all_dates:
                            min_date = parsed_col.min().date()
                            max_date = parsed_col.max().date()
                            start_date, end_date = st.date_input(
                                f"Select date range for `{col}`",
                                value=(min_date, max_date),
                                min_value=min_date,
                                max_value=max_date,
                                key=f"range_{col}"
                            )
                            
                            # Filter the datetime DataFrame using boolean indexing
                            date_mask = (datetime_df['__datetime__'].dt.date >= start_date) & \
                                      (datetime_df['__datetime__'].dt.date <= end_date)
                            filtered_df = datetime_df[date_mask].copy()
                        else:
                            filtered_df = datetime_df.copy()
                        
                        # Resample and count records
                        resampled = filtered_df.set_index('__datetime__').resample(freq_map[freq])
                        record_counts = resampled.size().rename("Total Records")
                        chart_df = pd.DataFrame(record_counts)
                        
                        if primary_keys:
                            try:
                                # Create a DataFrame with primary keys and datetime
                                pk_datetime_df = pd.DataFrame({
                                    '__datetime__': filtered_df['__datetime__']
                                })
                                
                                # Add primary keys by using the current dataframe
                                for pk in primary_keys:
                                    if pk in df.columns:
                                        # Get the values using the filtered indices
                                        if isinstance(filtered_df.index, pd.MultiIndex):
                                            # For multi-level index, get the first level
                                            idx = filtered_df.index.get_level_values(0)
                                        else:
                                            idx = filtered_df.index
                                        
                                        # Use loc to get values by index
                                        pk_values = df.loc[idx, pk].values
                                        pk_datetime_df[pk] = pk_values
                                
                                # Drop duplicates based on primary keys and resample
                                unique_keys_df = pk_datetime_df.drop_duplicates(subset=primary_keys)
                                unique_counts = unique_keys_df.set_index('__datetime__').resample(freq_map[freq]).size().rename("Unique Primary Keys")
                                chart_df = chart_df.join(unique_counts, how='outer').fillna(0)
                            except Exception as e:
                                st.warning(f"Could not calculate unique primary keys: {str(e)}")
                        
                        st.line_chart(chart_df)
                        continue
                    
                    # Display wordcloud if available
                    if 'wordcloud_text' in insights:
                        st.markdown("#### ☁️ Word Cloud (Long Text Field)")
                        try:
                            wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA').generate(insights['wordcloud_text'])
                            fig, ax = plt.subplots(figsize=(10, 5))
                            fig.patch.set_alpha(0)
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"⚠️ Word cloud generation failed: {str(e)}")
                    
                    # Display charts
                    for chart_type, chart_df in insights['charts']:
                        if chart_type == 'bar_chart':
                            chart = create_bar_chart(chart_df, 'Occurrences', col, "Top 10 Values (All Records)")
                            st.altair_chart(chart, use_container_width=True)
                        elif chart_type == 'bar_chart_pk':
                            chart = create_bar_chart(chart_df, 'Occurrences', col, "Top 10 Values (Per Unique Primary Key)", color_scheme='greens')
                            st.markdown("#### Top Values (Per Primary Key)")
                            st.altair_chart(chart, use_container_width=True)
                        elif chart_type == 'donut_chart':
                            fig = render_donut_chart(chart_df, 'Occurrences', col, "Value Distribution (All Records)")
                            st.plotly_chart(fig, use_container_width=True)
                        elif chart_type == 'donut_chart_pk':
                            fig = render_donut_chart(chart_df, 'Occurrences', col, "Value Distribution (Per Unique Primary Key)", color_scheme='greens')
                            st.markdown("#### Value Distribution (Per Primary Key)")
                            st.plotly_chart(fig, use_container_width=True)
                        elif chart_type == 'histogram':
                            hist = alt.Chart(chart_df).mark_bar(color='teal').encode(
                                alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col),
                                y=alt.Y('count()', title='Count')
                            ).properties(
                                width=600,
                                height=300,
                                title="Distribution"
                            )
                            st.altair_chart(hist, use_container_width=True)
                    
                    # Display tables
                    for table_type, table_df in insights['tables']:
                        if table_type == 'value_counts':
                            st.markdown("### 📊 Value Counts and Percentages")
                            st.dataframe(table_df, use_container_width=True)
                        elif table_type == 'value_counts_pk':
                            st.markdown("### 📊 Value Counts and Percentages (Per Primary Key)")
                            st.dataframe(table_df, use_container_width=True)
                        elif table_type == 'Unique Values':
                            with st.expander("📋 View Unique Values (with counts)"):
                                st.dataframe(table_df, use_container_width=True)
            
            # Clear the progress bar and status text
            progress_bar.empty()
            status_text.empty()
            
            # Show progress
            current_batch_size = len(current_batch)
            st.info(f"Showing fields {start_idx + 1} to {end_idx} of {total_fields} (Batch {st.session_state.current_batch + 1} of {(total_fields + st.session_state.batch_size - 1) // st.session_state.batch_size})")
            
            # Add navigation buttons with centered batch counter
            st.markdown("---")  # Add a separator
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.session_state.current_batch > 0:
                    if st.button("← Previous", use_container_width=True):
                        st.session_state.current_batch -= 1
                        st.rerun()
                else:
                    st.button("← Previous", use_container_width=True, disabled=True)
            with col2:
                total_batches = (total_fields + st.session_state.batch_size - 1) // st.session_state.batch_size
                current_page = st.session_state.current_batch + 1
                
                # Create a container for the page navigation
                with st.container():
                    # Add page number input with consistent height
                    new_page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_batches,
                        value=current_page,
                        key="page_input",
                        label_visibility="collapsed"
                    )
                    
                    # If page number changes, update the current batch
                    if new_page != current_page:
                        st.session_state.current_batch = new_page - 1
                        st.rerun()
                    
                    # Center the page counter text
                    st.markdown(
                        f"<div style='text-align: center; padding: 0.5rem;'><strong>{new_page} of {total_batches}</strong></div>",
                        unsafe_allow_html=True
                    )
            with col3:
                if end_idx < total_fields:
                    if st.button("Next →", use_container_width=True):
                        st.session_state.current_batch += 1
                        st.rerun()
                else:
                    st.button("Next →", use_container_width=True, disabled=True)
            st.markdown("---")  # Add a separator
        else:
            st.success("✅ All fields have been processed!")

    st.markdown('<div id="pattern-detection"></div>', unsafe_allow_html=True)
    st.markdown("## Pattern Detection")
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

    # Initialize session state for pattern detection
    if 'pattern_detection_run' not in st.session_state:
        st.session_state.pattern_detection_run = False

    if st.button("Run Pattern Detection"):
        st.session_state.pattern_detection_run = True
        st.rerun()

    if st.session_state.pattern_detection_run:
        # Create progress bar for pattern detection
        pattern_progress = st.progress(0)
        pattern_status = st.empty()

        # Process patterns in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pattern_futures = {
                executor.submit(process_patterns_parallel, df[col].dropna().astype(str), col): col 
                for col in df.columns
            }
            
            all_pattern_info = []
            total_columns = len(df.columns)
            completed = 0
            
            for future in as_completed(pattern_futures):
                col = pattern_futures[future]
                completed += 1
                pattern_progress.progress(completed / total_columns)
                pattern_status.text(f"Processing patterns for column {completed}/{total_columns}: {col}")
                
                try:
                    pattern_info = future.result()
                    all_pattern_info.extend(pattern_info)
                except Exception as e:
                    st.error(f"Error processing patterns for column {col}: {str(e)}")

        # Clear progress indicators
        pattern_progress.empty()
        pattern_status.empty()

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

    st.markdown('<div id="outlier-detection"></div>', unsafe_allow_html=True)
    st.markdown("## Outlier Detection")
    st.markdown("""
    This section helps identify outliers in your numerical data using two methods:
    1. **Z-score**: Identifies values that deviate more than 3 standard deviations from the mean
    2. **IQR (Interquartile Range)**: Identifies values outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    
    Outliers can indicate:
    - Data entry errors
    - Measurement errors
    - Natural variation
    - Special cases that need attention
    """)

    # Initialize session state for outlier detection
    if 'outlier_detection_run' not in st.session_state:
        st.session_state.outlier_detection_run = False

    if st.button("Run Outlier Detection"):
        st.session_state.outlier_detection_run = True
        st.rerun()

    if st.session_state.outlier_detection_run:
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numerical_cols:
            st.warning("⚠️ No numerical columns found for outlier detection.")
        else:
            # Create progress bar for outlier detection
            outlier_progress = st.progress(0)
            outlier_status = st.empty()

            # Process outliers in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                outlier_futures = {}
                for col in numerical_cols:
                    # Skip columns with too many nulls or all same values
                    if df[col].nunique() <= 1 or df[col].isnull().mean() > 0.5:
                        continue
                    
                    # Calculate Z-score and IQR outliers
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Get outlier indices
                    z_score_outliers = df[z_scores > 3].index
                    iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                    
                    # Create outlier summary
                    outlier_summary = {
                        'Column': col,
                        'Total Values': len(df[col]),
                        'Z-score Outliers': len(z_score_outliers),
                        'IQR Outliers': len(iqr_outliers),
                        'Z-score Outlier %': round(len(z_score_outliers) / len(df[col]) * 100, 2),
                        'IQR Outlier %': round(len(iqr_outliers) / len(df[col]) * 100, 2),
                        'Min Value': df[col].min(),
                        'Max Value': df[col].max(),
                        'Mean': df[col].mean(),
                        'Std Dev': df[col].std(),
                        'Q1': q1,
                        'Q3': q3,
                        'IQR': iqr
                    }
                    
                    outlier_futures[col] = outlier_summary

            # Create summary DataFrame
            outlier_summary_df = pd.DataFrame(list(outlier_futures.values()))
            
            # Clear progress indicators
            outlier_progress.empty()
            outlier_status.empty()
            
            # Display summary
            st.markdown("### 📊 Outlier Summary by Column")
            if not outlier_summary_df.empty:
                st.dataframe(outlier_summary_df, use_container_width=True)

                # Add insights section for basic outlier detection
                st.markdown("### 💡 Insights")
                
                # Calculate overall statistics
                if 'Z-score Outliers' in outlier_summary_df.columns and 'IQR Outliers' in outlier_summary_df.columns:
                    total_outliers_z = outlier_summary_df['Z-score Outliers'].sum()
                    total_outliers_iqr = outlier_summary_df['IQR Outliers'].sum()
                    total_values = outlier_summary_df['Total Values'].sum()
                    
                    # Generate insights
                    insights = []
                    
                    # Overall outlier percentage
                    z_score_percentage = (total_outliers_z / total_values) * 100
                    iqr_percentage = (total_outliers_iqr / total_values) * 100
                    
                    insights.append(f"📈 **Overall Outlier Analysis:**")
                    insights.append(f"- Z-score method identified {total_outliers_z} outliers ({z_score_percentage:.2f}% of data)")
                    insights.append(f"- IQR method identified {total_outliers_iqr} outliers ({iqr_percentage:.2f}% of data)")
                    
                    # Compare methods
                    if abs(z_score_percentage - iqr_percentage) > 5:
                        insights.append(f"\n⚠️ **Method Comparison:**")
                        insights.append(f"- There's a significant difference between Z-score and IQR methods")
                        insights.append(f"- This suggests your data might not be normally distributed")
                        insights.append(f"- Consider using the IQR method for more reliable results")
                    
                    # Column-specific insights
                    insights.append(f"\n🔍 **Column-specific Insights:**")
                    for _, row in outlier_summary_df.iterrows():
                        col_name = row['Column']
                        z_outliers = row.get('Z-score Outliers', 0)
                        iqr_outliers = row.get('IQR Outliers', 0)
                        z_percent = row.get('Z-score Outlier %', 0)
                        iqr_percent = row.get('IQR Outlier %', 0)
                        
                        if z_outliers > 0 or iqr_outliers > 0:
                            insights.append(f"\n**{col_name}:**")
                            if z_outliers > 0:
                                insights.append(f"- Has {z_outliers} Z-score outliers ({z_percent:.2f}%)")
                            if iqr_outliers > 0:
                                insights.append(f"- Has {iqr_outliers} IQR outliers ({iqr_percent:.2f}%)")
                            
                            # Add specific insights based on the data
                            if 'Std Dev' in row and 'Mean' in row and row['Std Dev'] > row['Mean']:
                                insights.append(f"- High variability: Standard deviation ({row['Std Dev']:.2f}) is greater than mean ({row['Mean']:.2f})")
                            if 'IQR' in row and 'Mean' in row and row['IQR'] > row['Mean']:
                                insights.append(f"- Wide spread: IQR ({row['IQR']:.2f}) is greater than mean ({row['Mean']:.2f})")
                    
                    # Display insights
                    st.markdown("\n".join(insights))
                else:
                    st.info("No outliers were detected in any of the columns using either Z-score or IQR methods.")
            else:
                st.info("No outliers were detected in any of the columns.")

            # Allow user to select a column for detailed analysis
            selected_col = st.selectbox(
                "Select a column for detailed outlier analysis",
                options=numerical_cols,
                key="outlier_column_selector"
            )

            if selected_col:
                # Calculate statistics for selected column
                z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
                q1 = df[selected_col].quantile(0.25)
                q3 = df[selected_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Get outlier data
                z_score_outliers = df[z_scores > 3]
                iqr_outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]

                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Distribution Plot with Z-score Outliers")
                    fig = go.Figure()
                    
                    # Add histogram
                    fig.add_trace(go.Histogram(
                        x=df[selected_col],
                        name='Distribution',
                        nbinsx=50,
                        opacity=0.7
                    ))
                    
                    # Add outlier points
                    fig.add_trace(go.Scatter(
                        x=z_score_outliers[selected_col],
                        y=[0] * len(z_score_outliers),
                        mode='markers',
                        name='Z-score Outliers',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='x'
                        )
                    ))
                    
                    fig.update_layout(
                        title=f'Distribution of {selected_col} with Z-score Outliers',
                        xaxis_title=selected_col,
                        yaxis_title='Count',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("#### 📊 Box Plot with IQR Outliers")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Box(
                        y=df[selected_col],
                        name=selected_col,
                        boxpoints='outliers',
                        marker=dict(
                            color='red',
                            size=8
                        )
                    ))
                    
                    fig.update_layout(
                        title=f'Box Plot of {selected_col}',
                        yaxis_title=selected_col,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                # Display outlier details
                st.markdown("#### 📋 Outlier Details")
                
                # Z-score outliers
                st.markdown("##### Z-score Outliers (|z| > 3)")
                if not z_score_outliers.empty:
                    st.dataframe(z_score_outliers[[selected_col]], use_container_width=True)
                else:
                    st.info("No Z-score outliers found.")

                # IQR outliers
                st.markdown("##### IQR Outliers")
                if not iqr_outliers.empty:
                    st.dataframe(iqr_outliers[[selected_col]], use_container_width=True)
                else:
                    st.info("No IQR outliers found.")

                # Export options
                with st.expander("📤 Export Outlier Results"):
                    if not z_score_outliers.empty or not iqr_outliers.empty:
                        # Combine both types of outliers
                        all_outliers = pd.concat([
                            z_score_outliers[[selected_col]].assign(Method='Z-score'),
                            iqr_outliers[[selected_col]].assign(Method='IQR')
                        ])
                        
                        csv = all_outliers.to_csv(index=False).encode()
                        st.download_button(
                            "📄 Download Outliers as CSV",
                            data=csv,
                            file_name=f"{selected_col}_outliers.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No outliers to export.")

    st.markdown('<div id="advanced-outlier"></div>', unsafe_allow_html=True)
    st.markdown("## Advanced Outlier Detection")
    st.markdown("""
    This section provides advanced outlier detection methods:
    1. **Isolation Forest**: Detects outliers by isolating observations in random forests
    2. **Local Outlier Factor (LOF)**: Identifies outliers based on local density deviation
    3. **DBSCAN**: Density-based clustering that can identify outliers as noise points
    4. **k-Means**: Identifies outliers as points far from cluster centers
    
    These methods are particularly useful for:
    - Multivariate outlier detection
    - Complex patterns and relationships
    - Non-linear data distributions
    """)

    # Initialize session state for advanced outlier detection
    if 'advanced_outlier_detection_run' not in st.session_state:
        st.session_state.advanced_outlier_detection_run = False

    if st.button("Run Advanced Outlier Detection"):
        st.session_state.advanced_outlier_detection_run = True
        st.rerun()

    if st.session_state.advanced_outlier_detection_run:
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numerical_cols) < 2:
            st.warning("⚠️ Need at least 2 numerical columns for advanced outlier detection.")
        else:
            # Allow user to select columns for analysis
            selected_cols = st.multiselect(
                "Select columns for advanced outlier detection (choose 2 or more)",
                options=numerical_cols,
                default=numerical_cols[:min(5, len(numerical_cols))],
                key="advanced_outlier_columns"
            )

            if len(selected_cols) >= 2:
                # Prepare data
                X = df[selected_cols].dropna()
                
                if len(X) == 0:
                    st.error("No valid data points after removing null values.")
                else:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Initialize results dictionary
                    results = {}

                    # 1. Isolation Forest
                    status_text.text("Running Isolation Forest...")
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    iso_scores = iso_forest.fit_predict(X)
                    results['Isolation Forest'] = {
                        'scores': iso_scores,
                        'outliers': X[iso_scores == -1]
                    }
                    progress_bar.progress(25)

                    # 2. Local Outlier Factor
                    status_text.text("Running Local Outlier Factor...")
                    from sklearn.neighbors import LocalOutlierFactor
                    lof = LocalOutlierFactor(contamination=0.1)
                    lof_scores = lof.fit_predict(X)
                    results['Local Outlier Factor'] = {
                        'scores': lof_scores,
                        'outliers': X[lof_scores == -1]
                    }
                    progress_bar.progress(50)

                    # 3. DBSCAN
                    status_text.text("Running DBSCAN...")
                    from sklearn.cluster import DBSCAN
                    from sklearn.preprocessing import StandardScaler
                    # Scale the data for DBSCAN
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    dbscan = DBSCAN(eps=0.5, min_samples=5)
                    dbscan_labels = dbscan.fit_predict(X_scaled)
                    results['DBSCAN'] = {
                        'scores': dbscan_labels,
                        'outliers': X[dbscan_labels == -1]
                    }
                    progress_bar.progress(75)

                    # 4. k-Means
                    status_text.text("Running k-Means...")
                    from sklearn.cluster import KMeans
                    from sklearn.metrics import silhouette_score
                    
                    # Find optimal k using silhouette score
                    silhouette_scores = []
                    K = range(2, min(11, len(X)))
                    for k in K:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        labels = kmeans.fit_predict(X_scaled)
                        if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
                            score = silhouette_score(X_scaled, labels)
                            silhouette_scores.append(score)
                    
                    optimal_k = K[np.argmax(silhouette_scores)] if silhouette_scores else 3
                    
                    # Run k-means with optimal k
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                    kmeans_labels = kmeans.fit_predict(X_scaled)
                    
                    # Calculate distances to cluster centers
                    distances = np.min(kmeans.transform(X_scaled), axis=1)
                    threshold = np.percentile(distances, 90)  # Mark top 10% as outliers
                    kmeans_outliers = distances > threshold
                    
                    results['k-Means'] = {
                        'scores': kmeans_labels,
                        'outliers': X[kmeans_outliers]
                    }
                    progress_bar.progress(100)

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Display results
                    st.markdown("### 📊 Outlier Detection Results")

                    # Create summary table
                    summary_data = []
                    for method, result in results.items():
                        n_outliers = len(result['outliers'])
                        summary_data.append({
                            'Method': method,
                            'Outliers Found': n_outliers,
                            'Outlier Percentage': f"{(n_outliers / len(X) * 100):.2f}%"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)

                    # Add insights section for advanced outlier detection
                    st.markdown("### 💡 Advanced Outlier Detection Insights")
                    
                    # Calculate overall statistics
                    total_outliers = sum(len(result['outliers']) for result in results.values())
                    total_points = len(X)
                    
                    # Generate insights
                    advanced_insights = []
                    
                    # Overall analysis
                    advanced_insights.append(f"📈 **Overall Analysis:**")
                    advanced_insights.append(f"- Total outliers detected: {total_outliers} ({total_outliers/total_points*100:.2f}% of data)")
                    
                    # Method comparison
                    advanced_insights.append(f"\n🔍 **Method Comparison:**")
                    method_counts = {method: len(result['outliers']) for method, result in results.items()}
                    max_method = max(method_counts.items(), key=lambda x: x[1])
                    min_method = min(method_counts.items(), key=lambda x: x[1])
                    
                    advanced_insights.append(f"- {max_method[0]} detected the most outliers ({max_method[1]})")
                    advanced_insights.append(f"- {min_method[0]} detected the least outliers ({min_method[1]})")
                    
                    # Consistency analysis
                    common_outliers = set()
                    for method, result in results.items():
                        if len(common_outliers) == 0:
                            common_outliers = set(result['outliers'].index)
                        else:
                            common_outliers &= set(result['outliers'].index)
                    
                    if len(common_outliers) > 0:
                        advanced_insights.append(f"\n✅ **High Confidence Outliers:**")
                        advanced_insights.append(f"- {len(common_outliers)} outliers were detected by all methods")
                        advanced_insights.append(f"- These are high-confidence outliers that should be investigated")
                    
                    # Method-specific insights
                    advanced_insights.append(f"\n📊 **Method-specific Insights:**")
                    for method, result in results.items():
                        n_outliers = len(result['outliers'])
                        if n_outliers > 0:
                            advanced_insights.append(f"\n**{method}:**")
                            advanced_insights.append(f"- Detected {n_outliers} outliers ({n_outliers/total_points*100:.2f}%)")
                            
                            # Add method-specific insights
                            if method == 'Isolation Forest':
                                advanced_insights.append(f"- Good for detecting global outliers in high-dimensional data")
                            elif method == 'Local Outlier Factor':
                                advanced_insights.append(f"- Effective at finding local outliers in clusters")
                            elif method == 'DBSCAN':
                                advanced_insights.append(f"- Identifies outliers as points that don't belong to any cluster")
                            elif method == 'k-Means':
                                advanced_insights.append(f"- Finds outliers as points far from cluster centers")
                    
                    # Recommendations
                    advanced_insights.append(f"\n💡 **Recommendations:**")
                    if total_outliers/total_points > 0.1:
                        advanced_insights.append(f"- High number of outliers detected. Consider:")
                        advanced_insights.append(f"  1. Checking for data quality issues")
                        advanced_insights.append(f"  2. Investigating the nature of these outliers")
                        advanced_insights.append(f"  3. Reviewing data collection methods")
                    else:
                        advanced_insights.append(f"- Moderate number of outliers detected. Consider:")
                        advanced_insights.append(f"  1. Reviewing the identified outliers")
                        advanced_insights.append(f"  2. Understanding their business context")
                        advanced_insights.append(f"  3. Deciding whether to keep or remove them")
                    
                    # Display insights
                    st.markdown("\n".join(advanced_insights))

                    # Visualize results
                    st.markdown("### 📈 Visualization")
                    
                    # Select visualization method
                    viz_method = st.selectbox(
                        "Choose visualization method",
                        ["2D Scatter Plot", "3D Scatter Plot"],
                        key="viz_method"
                    )

                    if viz_method == "2D Scatter Plot":
                        # Select columns for 2D plot
                        col1, col2 = st.columns(2)
                        with col1:
                            x_col = st.selectbox("Select X-axis", selected_cols, index=0)
                        with col2:
                            y_col = st.selectbox("Select Y-axis", selected_cols, index=1)

                        # Create plot for each method
                        for method, result in results.items():
                            fig = go.Figure()
                            
                            # Plot normal points
                            normal_points = X[result['scores'] != -1]
                            fig.add_trace(go.Scatter(
                                x=normal_points[x_col],
                                y=normal_points[y_col],
                                mode='markers',
                                name='Normal',
                                marker=dict(color='blue', size=8)
                            ))
                            
                            # Plot outliers
                            outliers = result['outliers']
                            fig.add_trace(go.Scatter(
                                x=outliers[x_col],
                                y=outliers[y_col],
                                mode='markers',
                                name='Outlier',
                                marker=dict(color='red', size=10, symbol='x')
                            ))
                            
                            fig.update_layout(
                                title=f'{method} Outlier Detection',
                                xaxis_title=x_col,
                                yaxis_title=y_col,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

                    else:  # 3D Scatter Plot
                        # Select columns for 3D plot
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_col = st.selectbox("Select X-axis", selected_cols, index=0)
                        with col2:
                            y_col = st.selectbox("Select Y-axis", selected_cols, index=1)
                        with col3:
                            z_col = st.selectbox("Select Z-axis", selected_cols, index=2)

                        # Create plot for each method
                        for method, result in results.items():
                            fig = go.Figure()
                            
                            # Plot normal points
                            normal_points = X[result['scores'] != -1]
                            fig.add_trace(go.Scatter3d(
                                x=normal_points[x_col],
                                y=normal_points[y_col],
                                z=normal_points[z_col],
                                mode='markers',
                                name='Normal',
                                marker=dict(color='blue', size=5)
                            ))
                            
                            # Plot outliers
                            outliers = result['outliers']
                            fig.add_trace(go.Scatter3d(
                                x=outliers[x_col],
                                y=outliers[y_col],
                                z=outliers[z_col],
                                mode='markers',
                                name='Outlier',
                                marker=dict(color='red', size=8, symbol='x')
                            ))
                            
                            fig.update_layout(
                                title=f'{method} Outlier Detection',
                                scene=dict(
                                    xaxis_title=x_col,
                                    yaxis_title=y_col,
                                    zaxis_title=z_col
                                ),
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

                    # Export results
                    with st.expander("📤 Export Results"):
                        # Combine all outlier results
                        all_outliers = pd.DataFrame()
                        for method, result in results.items():
                            outliers = result['outliers'].copy()
                            outliers['Method'] = method
                            all_outliers = pd.concat([all_outliers, outliers])
                        
                        if not all_outliers.empty:
                            csv = all_outliers.to_csv(index=False).encode()
                            st.download_button(
                                "📄 Download All Outliers as CSV",
                                data=csv,
                                file_name="advanced_outliers.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No outliers found to export.")

    if sidebar_visible:
        st.markdown('<div id="builtin-extraction"></div>', unsafe_allow_html=True)
        st.markdown("## Built-in Extraction Insights")

        # Initialize session state for extraction insights
        if 'extraction_insights_run' not in st.session_state:
            st.session_state.extraction_insights_run = False

        if st.button("Run Extraction Insights"):
            st.session_state.extraction_insights_run = True
            st.rerun()

        if st.session_state.extraction_insights_run:
            # Create progress bar for extractions
            extraction_progress = st.progress(0)
            extraction_status = st.empty()

            # Process extractions in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                extraction_futures = {
                    executor.submit(
                        process_extraction_parallel,
                        df[col].dropna(),
                        col,
                        len(df),
                        extract_country_region
                    ): col 
                    for col in df.select_dtypes(include=['object', 'string']).columns
                }
                
                all_extraction_data = []
                total_columns = len(df.select_dtypes(include=['object', 'string']).columns)
                completed = 0
                
                for future in as_completed(extraction_futures):
                    col = extraction_futures[future]
                    completed += 1
                    extraction_progress.progress(completed / total_columns)
                    extraction_status.text(f"Processing extractions for column {completed}/{total_columns}: {col}")
                    
                    try:
                        extraction_data = future.result()
                        if extraction_data:  # Only extend if we got results
                            all_extraction_data.extend(extraction_data)
                    except Exception as e:
                        st.error(f"Error processing extractions for column {col}: {str(e)}")

            # Clear progress indicators
            extraction_progress.empty()
            extraction_status.empty()

            # Process results by category
            summary_data = []
            region_summary_data = []
            compliance_summary_data = []
            business_unit_summary_data = []

            for data in all_extraction_data:
                if 'Countries Found' in data:
                    summary_data.append(data)
                if 'Regions Found' in data:
                    region_summary_data.append(data)
                if 'Compliance Found' in data:
                    compliance_summary_data.append(data)
                if 'Business Units Found' in data:
                    business_unit_summary_data.append(data)

            # Create DataFrames for summaries
            summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
            region_summary_df = pd.DataFrame(region_summary_data) if region_summary_data else pd.DataFrame()
            compliance_summary_df = pd.DataFrame(compliance_summary_data) if compliance_summary_data else pd.DataFrame()
            business_unit_summary_df = pd.DataFrame(business_unit_summary_data) if business_unit_summary_data else pd.DataFrame()

            # Display summaries
            if not summary_df.empty:
                st.write("### 🌍 Country Extraction Summary by Column")
                st.dataframe(summary_df)
                with st.expander("📤 Export Country Results"):
                    csv = summary_df.to_csv(index=False).encode()
                    st.download_button("📄 Download as CSV", data=csv, file_name="country_extraction.csv", mime="text/csv")
            else:
                st.write("No countries were extracted from the data.")

            if not region_summary_df.empty:
                st.write("### 🌐 Region Extraction Summary by Column")
                st.dataframe(region_summary_df)
                with st.expander("📤 Export Region Results"):
                    csv = region_summary_df.to_csv(index=False).encode()
                    st.download_button("📄 Download as CSV", data=csv, file_name="region_extraction.csv", mime="text/csv")
            else:
                st.write("No regions were extracted from the data.")

            if not compliance_summary_df.empty:
                st.write("### 📋 Compliance Extraction Summary by Column")
                st.dataframe(compliance_summary_df)
                with st.expander("📤 Export Compliance Results"):
                    csv = compliance_summary_df.to_csv(index=False).encode()
                    st.download_button("📄 Download as CSV", data=csv, file_name="compliance_extraction.csv", mime="text/csv")
            else:
                st.write("No compliance terms were extracted from the data.")

            if not business_unit_summary_df.empty:
                st.write("### 🏢 Business Unit Extraction Summary by Column")
                st.dataframe(business_unit_summary_df)
                with st.expander("📤 Export Business Unit Results"):
                    csv = business_unit_summary_df.to_csv(index=False).encode()
                    st.download_button("📄 Download as CSV", data=csv, file_name="business_unit_extraction.csv", mime="text/csv")
            else:
                st.write("No business units were extracted from the data.")

    # --- Custom Extraction Summary ---
    if "custom_categories" in st.session_state and st.session_state.custom_categories and len(st.session_state.custom_categories) > 0:
        st.markdown('<div id="custom-extraction-insights"></div>', unsafe_allow_html=True)
        st.markdown("## Custom Extraction Insights")
        
        # Initialize session state for custom extraction
        if 'custom_extraction_run' not in st.session_state:
            st.session_state.custom_extraction_run = False

        if st.button("Run Custom Extraction"):
            st.session_state.custom_extraction_run = True
            st.rerun()

        if st.session_state.custom_extraction_run:
            with st.expander("🔧 Debug: Current Custom Categories"):
                st.write("Active custom categories:", list(st.session_state.custom_categories.keys()))
                for cat, meta in st.session_state.custom_categories.items():
                    st.write(f"- {cat}: {len(meta['keywords'])} keywords")

            total_records = len(df)

            for category_name, meta in st.session_state.custom_categories.items():
                automaton = meta["automaton"]
                
                # Create progress bar for custom extraction
                custom_progress = st.progress(0)
                custom_status = st.empty()

                st.subheader(f"🔍 Extraction Summary for `{category_name}`")

                # Process custom extractions in parallel
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    custom_futures = {
                        executor.submit(
                            process_custom_extraction_parallel,
                            df[col].dropna(),
                            col,
                            total_records,
                            category_name,
                            automaton
                        ): col 
                        for col in df.select_dtypes(include=["object", "string"]).columns
                    }
                    
                    all_custom_data = []
                    total_columns = len(df.select_dtypes(include=["object", "string"]).columns)
                    completed = 0
                    
                    for future in as_completed(custom_futures):
                        col = custom_futures[future]
                        completed += 1
                        custom_progress.progress(completed / total_columns)
                        custom_status.text(f"Processing {category_name} extraction for column {completed}/{total_columns}: {col}")
                        
                        try:
                            custom_data = future.result()
                            all_custom_data.extend(custom_data)
                        except Exception as e:
                            st.error(f"Error processing custom extraction for column {col}: {str(e)}")

                # Clear progress indicators
                custom_progress.empty()
                custom_status.empty()

                summary_df = pd.DataFrame(all_custom_data)

                if not summary_df.empty:
                    st.markdown(f"### Summary Table for `{category_name}`")
                    st.dataframe(summary_df)
                else:
                    st.info(f"No `{category_name}` matches found.")

# Move Save Session section to the end of the file, after all other sections
if ('uploaded_file' in st.session_state and st.session_state.uploaded_file is not None) or \
   ('df' in locals() and df is not None) or \
   ('df' in st.session_state and st.session_state.df is not None):
    st.markdown('<div id="save-session"></div>', unsafe_allow_html=True)
    st.markdown("## Save Session")
    if st.button("💾 Save Session", key="save_session_button"):
        # Get the dataframe from either locals or session state
        current_df = df if 'df' in locals() and df is not None else st.session_state.df
        
        if current_df is not None:
            session_data = {
                "dataframe": current_df,
                "primary_keys": primary_keys if 'primary_keys' in locals() else None,
                "countries_input": st.session_state.get('countries_input', ""),
                "regions_input": st.session_state.get('regions_input', ""),
                "compliance_input": st.session_state.get('compliance_input', ""),
                "business_unit_input": st.session_state.get('business_unit_input', ""),
                "custom_categories": {
                    cat: data["keywords"] for cat, data in st.session_state.get("custom_categories", {}).items()
                }
            }

            # Prepare save directory
            save_dir = "EDA_Reports"
            os.makedirs(save_dir, exist_ok=True)

            # Build filename: <original_filename>_<timestamp>.pkl
            original_filename = st.session_state.file_name.rsplit('.', 1)[0] if st.session_state.file_name else "session"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_filename = f"{original_filename}_{timestamp}.pkl"
            save_path = os.path.join(save_dir, full_filename)

            # Save to file
            with open(save_path, "wb") as f:
                pickle.dump(session_data, f)

            st.success(f"✅ Session saved to `{save_path}`")
        else:
            st.warning("⚠️ No dataframe available to save.")

# Display total analysis time at the end
if 'analysis_start_time' in st.session_state:
    total_time = time.time() - st.session_state.analysis_start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    if minutes > 0:
        time_str = f"{minutes} minutes and {seconds:.2f} seconds"
    else:
        time_str = f"{seconds:.2f} seconds"
    st.success(f"⏱️ Total Analysis Time: {time_str}")


