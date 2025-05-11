# 📊 DataSleuth – Smart EDA Viewer

DataSleuth is a powerful, interactive Streamlit-based application for performing intelligent Exploratory Data Analysis (EDA) on CSV and Excel datasets. It supports primary key detection, field-wise summaries, trend analysis, pattern recognition, keyword extraction, and more.

---

## 🚀 Features

- 📂 Load sessions and upload CSV/XLSX files  
- 🧾 Field summaries with null/unique insights  
- 🔎 Primary key detection and validation  
- 📈 Trend charts for date/datetime fields  
- ☁️ Word clouds and value distribution visualizations  
- 🌍 Country/Region and custom keyword extraction  
- 🔍 Pattern detection with symbolic abstraction  
- 💾 Save and restore session states  
- ✅ Dynamic filtering and sidebar controls

---

## 📦 Requirements

### Local Setup
- Python 3.8+
- pip

### Docker Setup
- Docker Engine (v20+ recommended)

---

## 🛠️ Local Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/datasleuth.git
   cd datasleuth
   ```

2. **Create a virtual environment** _(optional but recommended)_:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:

   ```bash
   streamlit run app.py
   ```

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
├── app.py            # Main Streamlit application
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