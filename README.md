<div align="center">

# 🏠 SmartEstate: House Price Prediction System
### *An End-to-End Machine Learning Framework for Precision Real Estate Valuation*

---

[![Project Overview](https://img.shields.io/badge/📖_Overview-blue?style=for-the-badge)](#-project-overview)
[![Key Features](https://img.shields.io/badge/✨_Key_Features-6f42c1?style=for-the-badge)](#-key-features)
[![Tech Stack](https://img.shields.io/badge/🛠️_Tech_Stack-success?style=for-the-badge)](#-tech-stack)
[![Architecture](https://img.shields.io/badge/🏗️_Architecture-orange?style=for-the-badge)](#-technical-architecture)
[![Installation](https://img.shields.io/badge/🚀_Installation-red?style=for-the-badge)](#-installation--getting-started)
[![Contact](https://img.shields.io/badge/📩_Contact-lightgrey?style=for-the-badge)](#-contact)

---

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005850?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-4caf50?style=flat-square)](https://opensource.org/licenses/MIT)

**Empowering real estate decisions through advanced statistical intelligence.**

</div>

---

## 📖 Project Overview

The **SmartEstate Prediction System** is a robust, production-grade machine learning framework designed to eliminate subjectivity in property appraisal. Developed under the **Codiom** initiative, it provides a seamless pipeline from raw data ingestion to real-time API inference.

This framework manages the complete ML lifecycle:
1. **Ingestion:** Automated ETL from validated property datasets.
2. **Refinement:** High-performance feature engineering and statistical noise filtering.
3. **Intelligence:** Training of optimized regression architectures with cross-validation.
4. **Inference:** A modular FastAPI gateway for scalable production deployment.

---

## ✨ Key Features

* **🔍 Statistical EDA:** Automated multi-variate analysis to decode complex pricing patterns and feature importance.
* **🛠️ Advanced Preprocessing:**
    * **Categorical Encoding:** Seamless handling of spatial and structural qualitative data.
    * **Outlier Mitigation:** Implementation of **Interquartile Range (IQR)** filters to ensure training stability.
    * **Unit Standardization:** Feature scaling via `StandardScaler` for optimized hyperparameter convergence.
* **🤖 Optimization Engine:** Utilizing **Ridge** and **Lasso** regularization techniques to maximize generalized accuracy and prevent overfitting.
* **💾 Enterprise Persistence:** Multi-artifact serialization ensures data transformers and model weights remain perfectly synced.
* **🚀 RESTful API Gateway:** High-speed inference delivery powered by **FastAPI** with automatic Pydantic validation.

---

## 🛠️ Tech Stack

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Development** | **Python 3.9+** | Core system logic and async API management. |
| **Data Engineering** | **Pandas / NumPy** | Vectorized manipulation and complex ETL. |
| **ML Engine** | **Scikit-Learn** | Pipeline orchestration and model training. |
| **Visualizations** | **Plotly / Seaborn** | Dynamic residual plots and correlation matrices. |
| **Deployment** | **FastAPI / Uvicorn** | High-concurrency production model serving. |

---

## 🏗️ Technical Architecture

The system implements a decoupled **Pipeline Architecture**, ensuring that data processing logic is strictly separated from the inference engine.

### Mathematical Validation

Model performance is evaluated using high-precision statistical metrics:

* **Coefficient of Determination ($R^2$):** Accuracy of variance explanation.
* **Root Mean Squared Error (RMSE):**
  $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **Mean Absolute Error (MAE):**
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

---

## 📂 Project Structure

```bash
.
├── 📁 data/
│   └── Housing.csv          # Curated property data
├── 📄 main.py               # ML Pipeline (Build -> Train -> Export)
├── 📄 app.py                # FastAPI Production Gateway
├── 📁 models/
│   ├── housing_model.pkl    # Serialized weights
│   ├── feature_names.pkl    # Schema metadata
│   └── scaler.pkl           # Pre-fitted parameters
├── 📄 requirements.txt      # Dependency manifest
└── 📄 README.md             # System Documentation
```

## 🚀 Installation & Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone [https://github.com/beraterolcelik/house-price-prediction.git](https://github.com/beraterolcelik/house-price-prediction.git)
cd house-price-prediction

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

```

### 2. Dependency Injection

```bash
pip install -r requirements.txt

```

### 3. Execution Flow
Build & Train Model:
```bash
python main.py

```
Start Production API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Access API Docs at: http://localhost:8000/docs

## 🗺️ Roadmap

- [ ] **Gradient Boosting**: Integrating XGBoost/LightGBM for superior capture of complex non-linear property patterns.
- [ ] **Cloud Readiness**: Full containerization via Docker and Kubernetes orchestration for global scalability.
- [ ] **Real-time Drift**: Automated monitoring systems to detect model performance decay in shifting markets.
- [ ] **Geospatial Insights**: Integration of Mapbox API for interactive, location-based property valuation heatmaps.

---

<div align="center" id="contact">

Architected with precision by Berat Erol Çelik Founder of Codiom

Software Engineering @ Istanbul Aydın University

</div>


















