import streamlit as st

st.set_page_config(
    page_title="Help - DataSleuth",
    page_icon="static/favicon_io/favicon-32x32.png",
    layout="wide"
)

st.title("❓ Help & Documentation")

# Introduction
st.header("Welcome to DataSleuth")
st.markdown("""
This guide will help you understand how to use DataSleuth effectively for your data analysis needs. 
From loading data to performing advanced analysis and saving sessions, we've got you covered.
""")

# Data Loading Section
st.header("📂 Loading Data")
st.markdown("""
### Getting Started
You can begin your analysis in two ways:
- Upload a file (supports CSV, JSON, or XML formats)
- Connect to a database (currently supports Spark Thrift Server)

### Database Connection Guide
1. **Create a Connection Profile**
   - Enter IP, port, and authentication details
   - Save the profile for future use
2. **Query the Saved Profile**
   - Use the saved profile to execute SQL queries
   - Fetch your data using the established connection

> ⚠️ **Important Note**: Always refresh the app before loading a new dataset to prevent unexpected behavior.
""")

# Table of Contents
st.header("📑 Table of Contents")
st.markdown("""
After uploading your data:
1. Click the "Refresh TOC" button
2. This generates an accurate and updated Table of Contents
3. Ensures all fields and insights are properly indexed
""")

# Extraction Features
st.header("🧠 Data Extraction")
st.markdown("""
### Built-in Extraction
- Enabled by default for each session
- Customizable from the sidebar
- Changes apply only to the current session

### Custom Extraction
Create custom rules in the "Add Custom Extraction" section:
1. Provide a name for the extraction
2. Enter comma-separated values to extract
3. Perfect for tagging business terms, categories, or patterns
""")

# Primary Key
st.header("🔑 Primary Key Management")
st.markdown("""
- Primary keys are automatically identified during data load
- Manual changes possible in the Primary Key Identification section
- If no valid key is found/selected, each record is treated as unique
""")

# Field Insights
st.header("🔍 Field Insights")
st.markdown("""
### Performance Optimization
For large datasets, you can:
- Paginate through fields
- Select specific fields for inspection
- This ensures faster rendering and targeted exploration
""")

# Computation Features
st.header("⚡ Trigger-Based Computation")
st.markdown("""
The following modules run only when explicitly triggered:
- Pattern Detection
- Outlier Detection
- Advanced Outlier Detection
- Built-in Extraction Insights
- Custom Extraction Insights

> This design prevents unnecessary computations and improves performance on large datasets.
""")

# Session Management
st.header("💾 Session Management")
st.markdown("""
### Saving Sessions
1. Click "Save Session" to store current analysis
2. Sessions are saved in the DataSleuth/EDA_Reports directory
3. Use "Load Previous Session" from the top menu to resume work
""")

# Advanced Filtering
st.header("🎯 Advanced Filtering")
st.markdown("""
### Available Filter Types
#### Text Fields
- Equals
- Contains
- Regex
- Starts With

#### Numeric Fields (Integer/Float)
- Less Than
- Greater Than
- Range

### How to Apply Filters
1. Select fields to filter on
2. Click "Apply Filters" to enable filtering mode
3. Choose filter type for each selected field
4. Enter filter values
5. Click "Apply Filters" again to execute
""")

# About Developer
st.header("👨‍💻 About Developer")
st.markdown("""
DataSleuth is developed and maintained by Jibin George.

- GitHub: [@J4ckFr05t/DataSleuth](https://github.com/J4ckFr05t/DataSleuth)
- Website: [jibingeorge.org](https://jibingeorge.org)

Feel free to contribute to the project or report any issues on GitHub!
""") 