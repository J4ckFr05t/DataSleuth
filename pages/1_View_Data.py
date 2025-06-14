import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Raw Data - DataSleuth",
    page_icon="📊",
    layout="wide"
)

st.title("🗃️ Raw Data View")

# Check if data is loaded in session state
if 'df' not in st.session_state:
    st.warning("⚠️ No data has been loaded yet. Please load data in the main page first.")
else:
    # Use filtered data if available, otherwise use original data
    df = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.df
    
    # Display basic information about the dataset in a stock market style
    st.markdown("### 📈 Dataset Statistics")
    
    # Create three columns for the stats
    col1, col2, col3 = st.columns(3)
    
    # Calculate memory usage in MB
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Style for the metric cards
    metric_style = """
    <style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #E0E0E0;
    }
    .metric-label {
        font-size: 14px;
        color: #888888;
        margin-top: 5px;
    }
    </style>
    """
    st.markdown(metric_style, unsafe_allow_html=True)
    
    # Display metrics in cards
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns):,}</div>
            <div class="metric-label">Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{memory_usage:.2f} MB</div>
            <div class="metric-label">Memory Usage</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a separator
    st.markdown("---")
    
    # Display active filters if any
    if st.session_state.active_filters:
        st.markdown("### 🔍 Active Filters")
        for field, filter_config in st.session_state.active_filters.items():
            try:
                if filter_config['type'] == 'equals':
                    st.markdown(f"- **{field}**: Equals {', '.join(map(str, filter_config['values']))}")
                elif filter_config['type'] == 'contains':
                    st.markdown(f"- **{field}**: Contains '{filter_config['value']}'")
                elif filter_config['type'] == 'starts_with':
                    st.markdown(f"- **{field}**: Starts with '{filter_config['value']}'")
                elif filter_config['type'] == 'regex':
                    st.markdown(f"- **{field}**: Matches pattern '{filter_config['value']}'")
                elif filter_config['type'] == 'range':
                    st.markdown(f"- **{field}**: Between {filter_config['min']} and {filter_config['max']}")
                elif filter_config['type'] == 'less_than':
                    st.markdown(f"- **{field}**: Less than {filter_config['value']}")
                elif filter_config['type'] == 'greater_than':
                    st.markdown(f"- **{field}**: Greater than {filter_config['value']}")
            except Exception as e:
                st.sidebar.error(f"Error displaying filter for {field}")
    
    # Display the data
    st.markdown("### 📋 Data Preview")
    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )
    
    # Add export option
    st.markdown("### 📤 Export Option")
    # CSV Export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download as CSV",
        csv,
        "raw_data.csv",
        "text/csv",
        key='download-csv'
    ) 