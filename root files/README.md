# 🏥 HealthGuard AI — Predictive Health Risk Analysis

A full-stack machine learning application for predicting patient health risks across 5 disease categories.

## 🧠 ML Models (5 Trained Models)

| Disease | Algorithm | Accuracy |
|---|---|---|
| Diabetes | Gradient Boosting | ~87.6% |
| Heart Disease | Gradient Boosting | ~85.9% |
| Hypertension | Random Forest | ~86.3% |
| Stroke | Gradient Boosting | ~83.2% |
| Obesity | Random Forest | ~90.7% |

**Features used (17):** Age, BMI, Blood Pressure, Blood Glucose, Cholesterol, Heart Rate, Smoking, Alcohol, Physical Activity, Diet Quality, Sleep, Stress, Family History (4 conditions)

---

## 📁 Project Structure

```
health_risk_app/
├── backend/
│   ├── app.py              # Flask REST API (main server)
│   ├── train_models.py     # ML model training script
│   └── db_mysql.py         # MySQL connector (production)
├── frontend/
│   ├── index.html          # SPA frontend
│   └── static/
│       ├── css/style.css   # Dark medical UI
│       └── js/app.js       # Vanilla JS frontend logic
├── ml_models/              # Saved model files (.pkl)
├── database/
│   └── schema.sql          # MySQL schema
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train ML models (first time only)
```bash
cd health_risk_app
python backend/train_models.py
```

### 3. Run the server
```bash
python backend/app.py
```

### 4. Open browser
```
http://localhost:5000
```

---

## 🗄 MySQL Setup (Production)

### Create database
```bash
mysql -u root -p < database/schema.sql
```

### Configure connection
```bash
cp .env.example .env
# Edit .env with your MySQL credentials
```

### Switch app.py to MySQL
In `backend/app.py`, import and use `db_mysql.py` functions:
```python
from db_mysql import init_db, db_get_patients, db_create_patient, db_save_assessment
init_db(app)
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server health & loaded models |
| GET | `/api/patients` | List all patients |
| POST | `/api/patients` | Register new patient |
| GET | `/api/patients/:id` | Get patient + history |
| POST | `/api/assess` | Run ML risk assessment |
| GET | `/api/dashboard/stats` | Dashboard analytics |
| GET | `/api/patients/:id/history` | Assessment history |

### Assessment Request Body
```json
{
  "patient_id": "P001",        // optional
  "age": 45,
  "bmi": 27.5,
  "blood_pressure_systolic": 135,
  "blood_pressure_diastolic": 85,
  "blood_glucose": 105,
  "cholesterol": 215,
  "heart_rate": 78,
  "smoking": 0,
  "alcohol_consumption": "Occasional",
  "physical_activity": "Light",
  "diet_quality": "Fair",
  "sleep_hours": 6.5,
  "stress_level": 7,
  "family_history_diabetes": 1,
  "family_history_heart": 0,
  "family_history_hypertension": 1,
  "family_history_cancer": 0
}
```

### Assessment Response
```json
{
  "success": true,
  "data": {
    "predictions": {
      "diabetes": { "probability": 52.3, "level": "High" },
      "heart_disease": { "probability": 38.1, "level": "Moderate" },
      ...
    },
    "overall_risk_score": 45.2,
    "risk_category": "Moderate",
    "recommendations": [...],
    "alerts": [...]
  }
}
```

---

## 🖥 Frontend Features

- **Dashboard** — Population analytics, risk distribution, disease rates, recent assessments
- **Risk Assessment** — 17-variable form → instant ML predictions with gauges
- **Patient Registry** — CRUD operations, assessment history
- **Register Patient** — Add new patients to the system

---

## ⚙ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, Vanilla JS, CSS3 (dark theme) |
| Backend | Python 3, Flask |
| ML | scikit-learn (GBM + Random Forest) |
| Database | MySQL 8+ (in-memory fallback for demo) |
| Fonts | Bebas Neue, DM Sans, Space Mono |