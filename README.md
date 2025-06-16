# DataSleuth – Smart EDA Viewer

<div align="center">
  <img src="static/logo.png" alt="DataSleuth Logo" width="200"/>
</div>

DataSleuth is a powerful, interactive Streamlit-based application for performing intelligent Exploratory Data Analysis (EDA) on CSV, Excel, JSON, and XML datasets. It supports advanced data analysis, visualization, and pattern detection capabilities.

---

## 🚀 Features

- 📂 Multi-format Support
  - CSV, Excel, JSON, and XML file upload
  - Database connection support (Spark Thrift Server)
  - Session state management

- 🔍 Advanced Data Analysis
  - Primary key detection and validation
  - Field-wise summaries with null/unique insights
  - Pattern detection with symbolic abstraction
  - Outlier detection using multiple algorithms (Z-score, IQR, Isolation Forest, DBSCAN, k-Means)
  - Advanced multi-dimensional outlier analysis

- 📊 Visualization Capabilities
  - Trend charts for date/datetime fields
  - Word clouds for text fields
  - Value distribution visualizations
  - Interactive data filtering and exploration

- 🌍 Smart Extraction Features
  - Country and region detection
  - Business unit identification
  - Compliance term extraction
  - Custom keyword extraction

- ⚙️ Performance Optimizations
  - Parallel processing for large datasets
  - Batch processing for field analysis
  - Progress tracking and status updates
  - Memory-efficient data handling

- 🛠️ User Experience
  - Dynamic filtering and sidebar controls
  - Paginated field analysis
  - Field-specific detailed views
  - Save and restore session states

---

## 📦 Requirements

### Local Setup
- Python 3.8+
- pip

### Docker Setup
- Docker Engine (v20+ recommended)

---

## 🛠️ Local Installation

- Windows: Double click DataSleuth.bat
- Linux/MacOS: chmod +x DataSleuth.sh; ./DataSleuth.sh

---

## 🐳 Docker Setup

1. **Build the Docker image**:

   ```bash
   docker build -t datasleuth .
   ```

2. **Run the container**:

   ```bash
   docker run -d -p 8501:8501 datasleuth
   ```

   Now visit: [http://localhost:8501](http://localhost:8501)

> **Note**: You can mount a volume for file persistence if you want to save sessions:
>
> ```bash
> docker run -d -p 8501:8501 -v $PWD/EDA_Reports:/app/EDA_Reports datasleuth
> ```

---

## 📁 File Structure

```
.
├── Home.py            # Main Streamlit application
├── Dockerfile        # Docker build instructions
├── README.md         # You're here
└── requirements.txt  # Python dependencies (create if missing)
```
---

## 🤝 Contributing

Pull requests and issues are welcome. If you'd like to contribute features like authentication, database support, or more visualizations, feel free to fork the project!

---

## 🛡️ License

MIT License. Feel free to use and modify.
