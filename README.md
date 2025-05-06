# 🧠 DataSleuth: Smart EDA Viewer

DataSleuth is a powerful, user-friendly Exploratory Data Analysis (EDA) tool built with Streamlit. It helps you explore large CSV/Excel files with:
- Field-wise summaries
- Value distribution charts
- Pattern detection (e.g., IP, email, Aa9-style formats)
- Primary key checks
- Country/region entity recognition
- Exportable results for Confluence or CSV
- Docker support for easy deployment

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/J4ckFr05t/DataSleuth.git
cd DataSleuth
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

### 1. Build the Docker image

```bash
docker build -t datasleuth .
```

### 2. Run the container

```bash
docker run -p 8501:8501 datasleuth
```

Then visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Configuration

The `.streamlit/config.toml` file controls Streamlit behavior like:
- Max file upload size

### Example: Enable dark mode and increase upload size

```toml
[server]
maxUploadSize = 200
```

> 💡 Max file size is in **MB**. Adjust `maxUploadSize` based on your dataset size.

---

## 📦 Export Features

You can export:
- Field summaries as CSV
- Confluence-compatible HTML
- Pattern detection results

---

## 🧠 Pattern Detection

DataSleuth auto-detects:
- Known formats: IP address, MAC, email, DNS, FQDN, phone numbers
- Abstract patterns: e.g., `Aa9@-` where:
  - `A` = Uppercase letter  
  - `a` = Lowercase letter  
  - `9` = Digit  
  - `@#-` = Special characters  
- Fields are **flagged** if fewer than **80%** of values follow the dominant pattern

---

## ✨ Example Output

- Interactive summary tables
- Bar charts for top values
- Histogram for numeric distributions
- Detected patterns per column with export

---

## 🤝 Contributing

Feel free to fork and improve! PRs welcome.

---

## 🛡️ License

MIT License
