import streamlit as st

st.set_page_config(
    page_title="Help - DataSleuth",
    page_icon="❓",
    layout="wide"
)

st.title("❓ Help & Documentation")

st.markdown("""
This section provides guidance on how to use DataSleuth effectively — from loading data to performing advanced analysis and saving sessions.

📂 Loading Data
You can begin analysis by:

Uploading a file in CSV, JSON, or XML format

Or connecting to a database (currently only Spark Thrift Server is supported)

🔌 Database Connection Instructions
To load data from a database:

Create a Connection Profile

Enter IP, port, and authentication details

Save this profile for future use

Query the Saved Profile

After saving, provide a SQL query to fetch the data using that profile

⚠️ Important:
If you want to load a new dataset, refresh the app before uploading or connecting. Skipping this may cause unexpected behavior.

📑 Table of Contents (TOC)
After uploading data, click the Refresh TOC button to generate an accurate and updated Table of Contents. This ensures all fields and insights are correctly indexed.

🧠 Extraction Settings
✅ Built-in Extraction
Enabled by default for each session. You can change or disable these from the sidebar.
Changes apply only to the current session.

✍️ Custom Extraction
Custom rules can be created in the "Add Custom Extraction" section:

Provide a name for the extraction

Enter comma-separated values to be extracted

These are useful for tagging business terms, categories, or patterns of interest

🔑 Primary Key Identification
A primary key is auto-identified during data load

You can manually change the primary key in the Primary Key Identification section

If no valid key is found or selected, DataSleuth will assume each record is unique

🔍 Per Field Insights
To optimize performance on large datasets:

You can paginate through fields

Or select only specific fields to inspect

This ensures faster rendering and targeted exploration.

⚡ Trigger-Based Computation
The following modules only run when explicitly triggered using their respective "Run" buttons:

Pattern Detection

Outlier Detection

Advanced Outlier Detection

Built-in Extraction Insights

Custom Extraction Insights

This design prevents unnecessary computations and improves performance on large datasets.

💾 Saving & Loading Sessions
Click "Save Session" to store the current analysis

Sessions are saved inside the DataSleuth/EDA_Reports directory

To resume work later, use "Load Previous Session" from the top menu

🎯 Advanced Filtering
DataSleuth provides robust, field-specific filtering:

Text Fields:

Equals

Contains

Regex

Starts With

Numeric Fields (Integer/Float):

Less Than

Greater Than

Range

🔁 How to Apply Filters:
Select the fields you want to filter on

Click "Apply Filters" to enable filtering mode

Choose filter type for each selected field

Enter filter values

Click "Apply Filters" again to execute
""") 