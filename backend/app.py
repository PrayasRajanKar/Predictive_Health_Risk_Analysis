"""
Predictive Health Risk Analysis — Flask Backend v3.0
✅ MySQL persistence — all data saved to database
✅ No hardcoded sample data — only real user-submitted data
✅ Syncs from MySQL on startup — survives server restarts
✅ 5 ML models: diabetes, heart_disease, hypertension, stroke, obesity
✅ Auto-trains models if .pkl files not found
✅ Full REST API with CORS
✅ Serves frontend from /frontend folder
"""

import os
import json
import numpy as np
import joblib
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)

FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

DISEASES = ['diabetes', 'heart_disease', 'hypertension', 'stroke', 'obesity']
MODELS   = {}

# ─────────────────────────────────────────────────────────────
# MYSQL CONFIGURATION  ← update password here
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    'host':     os.environ.get('DB_HOST',     'localhost'),
    'user':     os.environ.get('DB_USER',     'root'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': os.environ.get('DB_NAME',     'health_risk_db'),
    'port':     int(os.environ.get('DB_PORT',  3306)),
}

def get_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"  [DB ERROR] {e}")
        return None

# ─────────────────────────────────────────────────────────────
# IN-MEMORY CACHE  (loaded from MySQL on startup)
# ─────────────────────────────────────────────────────────────
patients_db        = {}   # { patient_id: dict }
assessments_db     = []   # [ dict, ... ]
assessment_counter = [1]

def sync_from_db():
    """Pull existing rows from MySQL into the in-memory cache on startup."""
    conn = get_db()
    if not conn:
        print("  [WARN] Cannot connect to DB — starting with empty cache.")
        return
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT * FROM patients ORDER BY created_at ASC")
        for row in cur.fetchall():
            # serialize any datetime objects
            row = {k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k, v in row.items()}
            patients_db[row['patient_id']] = row
        print(f"  Synced {len(patients_db)} patient(s) from DB")

        cur.execute("SELECT * FROM risk_predictions ORDER BY prediction_date ASC")
        rows = cur.fetchall()
        for row in rows:
            row = {k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k, v in row.items()}
            assessments_db.append(row)
            assessment_counter[0] = max(assessment_counter[0], int(row['id']) + 1)
        print(f"  Synced {len(assessments_db)} assessment(s) from DB")

    except Error as e:
        print(f"  [SYNC ERROR] {e}")
    finally:
        cur.close()
        conn.close()

# ─────────────────────────────────────────────────────────────
# ML — AUTO-TRAIN IF MODELS MISSING
# ─────────────────────────────────────────────────────────────
def generate_training_data(disease, n=6000):
    np.random.seed(42)
    age        = np.random.randint(18, 90, n).astype(float)
    bmi        = np.clip(np.random.normal(26, 5, n), 15, 55)
    bp_sys     = np.clip(np.random.normal(120, 20, n), 80, 200)
    bp_dia     = np.clip(np.random.normal(80, 12, n), 50, 130)
    glucose    = np.clip(np.random.normal(95, 25, n), 60, 400)
    cholest    = np.clip(np.random.normal(190, 40, n), 100, 400)
    heart_rate = np.clip(np.random.normal(75, 12, n), 45, 130)
    smoking    = np.random.randint(0, 2, n).astype(float)
    alcohol    = np.random.randint(0, 4, n).astype(float)
    activity   = np.random.randint(0, 4, n).astype(float)
    diet       = np.random.randint(0, 4, n).astype(float)
    sleep      = np.clip(np.random.normal(7, 1.5, n), 3, 12)
    stress     = np.clip(np.random.normal(5, 2, n), 1, 10)
    fam_diab   = np.random.randint(0, 2, n).astype(float)
    fam_heart  = np.random.randint(0, 2, n).astype(float)
    fam_hyp    = np.random.randint(0, 2, n).astype(float)
    fam_cancer = np.random.randint(0, 2, n).astype(float)

    X = np.column_stack([age, bmi, bp_sys, bp_dia, glucose, cholest,
                         heart_rate, smoking, alcohol, activity, diet,
                         sleep, stress, fam_diab, fam_heart, fam_hyp, fam_cancer])

    risk_map = {
        'diabetes':      0.30*(glucose>126) + 0.20*(bmi>30)     + 0.15*fam_diab    + 0.10*(age>45)      + 0.10*(activity<2) + 0.08*smoking      + 0.07*(diet<2),
        'heart_disease': 0.25*(cholest>240) + 0.20*(bp_sys>140) + 0.15*smoking      + 0.15*fam_heart     + 0.10*(age>50)     + 0.08*(glucose>126) + 0.07*(heart_rate>90),
        'hypertension':  0.30*(bp_sys>140)  + 0.20*fam_hyp      + 0.15*(bmi>30)    + 0.10*(age>40)      + 0.10*smoking      + 0.10*(alcohol>2)   + 0.05*(stress>7),
        'stroke':        0.25*(bp_sys>150)  + 0.20*(age>60)     + 0.15*fam_heart   + 0.15*(cholest>240) + 0.10*smoking      + 0.10*(glucose>126) + 0.05*(stress>8),
        'obesity':       0.35*(bmi>30)      + 0.20*(activity<1) + 0.15*(diet<2)    + 0.10*(sleep<6)     + 0.10*(stress>7)   + 0.10*(age>40),
    }
    noise = np.random.normal(0, 0.04, n)
    y = ((risk_map[disease] + noise).clip(0, 1) > 0.35).astype(int)
    return X, y


def train_and_save(disease):
    print(f"  Training: {disease}...")
    X, y   = generate_training_data(disease)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_tr)
    model  = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08,
                                         max_depth=4, random_state=42)
    model.fit(X_sc, y_tr)
    joblib.dump(model,  os.path.join(MODEL_DIR, f'{disease}_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'{disease}_scaler.pkl'))
    return model, scaler


def load_models():
    for disease in DISEASES:
        m_path = os.path.join(MODEL_DIR, f'{disease}_model.pkl')
        s_path = os.path.join(MODEL_DIR, f'{disease}_scaler.pkl')
        if os.path.exists(m_path) and os.path.exists(s_path):
            MODELS[disease] = {'model': joblib.load(m_path), 'scaler': joblib.load(s_path)}
            print(f"  Loaded:  {disease}")
        else:
            model, scaler = train_and_save(disease)
            MODELS[disease] = {'model': model, 'scaler': scaler}

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
ALCOHOL_MAP  = {'None': 0, 'Occasional': 1, 'Moderate': 2, 'Heavy': 3}
ACTIVITY_MAP = {'Sedentary': 0, 'Light': 1, 'Moderate': 2, 'Active': 3}
DIET_MAP     = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}

def build_feature_vector(data):
    return np.array([[
        float(data.get('age', 35)),
        float(data.get('bmi', 25)),
        float(data.get('blood_pressure_systolic', 120)),
        float(data.get('blood_pressure_diastolic', 80)),
        float(data.get('blood_glucose', 90)),
        float(data.get('cholesterol', 180)),
        float(data.get('heart_rate', 72)),
        int(bool(data.get('smoking', 0))),
        ALCOHOL_MAP.get(data.get('alcohol_consumption', 'None'), 0),
        ACTIVITY_MAP.get(data.get('physical_activity', 'Sedentary'), 0),
        DIET_MAP.get(data.get('diet_quality', 'Fair'), 1),
        float(data.get('sleep_hours', 7)),
        float(data.get('stress_level', 5)),
        int(bool(data.get('family_history_diabetes', 0))),
        int(bool(data.get('family_history_heart', 0))),
        int(bool(data.get('family_history_hypertension', 0))),
        int(bool(data.get('family_history_cancer', 0))),
    ]])

# ─────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────
def predict_risks(fv):
    results = {}
    for disease, bundle in MODELS.items():
        X_sc = bundle['scaler'].transform(fv)
        prob = float(bundle['model'].predict_proba(X_sc)[0][1])
        pct  = round(prob * 100, 1)
        results[disease] = {
            'probability': pct,
            'level': 'Critical' if pct >= 75 else 'High' if pct >= 50 else 'Moderate' if pct >= 25 else 'Low'
        }
    return results

def calculate_overall_score(predictions):
    w = {'heart_disease': 0.30, 'diabetes': 0.25, 'hypertension': 0.20, 'stroke': 0.15, 'obesity': 0.10}
    return round(sum(predictions[d]['probability'] * wt for d, wt in w.items()), 1)

def risk_category(score):
    return 'Critical' if score >= 75 else 'High' if score >= 50 else 'Moderate' if score >= 25 else 'Low'

def generate_risk_factors(data, predictions):
    factors = []
    bmi    = float(data.get('bmi', 25))
    bp_sys = float(data.get('blood_pressure_systolic', 120))
    gluc   = float(data.get('blood_glucose', 90))
    chol   = float(data.get('cholesterol', 180))
    age    = float(data.get('age', 35))
    stress = float(data.get('stress_level', 5))
    sleep  = float(data.get('sleep_hours', 7))

    if bp_sys >= 140:   factors.append({'name': 'High Blood Pressure',    'value': f'{bp_sys} mmHg', 'impact': 88, 'level': 'High'})
    elif bp_sys >= 130: factors.append({'name': 'Elevated Blood Pressure', 'value': f'{bp_sys} mmHg', 'impact': 55, 'level': 'Medium'})
    if bmi >= 30:       factors.append({'name': 'Obesity (BMI)',           'value': f'{bmi:.1f}',     'impact': 80, 'level': 'High'})
    elif bmi >= 25:     factors.append({'name': 'Overweight (BMI)',        'value': f'{bmi:.1f}',     'impact': 42, 'level': 'Medium'})
    if gluc >= 126:     factors.append({'name': 'High Blood Glucose',      'value': f'{gluc} mg/dL',  'impact': 85, 'level': 'High'})
    elif gluc >= 100:   factors.append({'name': 'Elevated Blood Glucose',  'value': f'{gluc} mg/dL',  'impact': 48, 'level': 'Medium'})
    if chol >= 240:     factors.append({'name': 'High Cholesterol',        'value': f'{chol} mg/dL',  'impact': 72, 'level': 'High'})
    elif chol >= 200:   factors.append({'name': 'Borderline Cholesterol',  'value': f'{chol} mg/dL',  'impact': 38, 'level': 'Medium'})
    if data.get('smoking'):
        factors.append({'name': 'Active Smoker',        'value': 'Yes',       'impact': 78, 'level': 'High'})
    if age >= 60:       factors.append({'name': 'Age Risk', 'value': f'{int(age)} yrs', 'impact': 65, 'level': 'High'})
    elif age >= 45:     factors.append({'name': 'Age Risk', 'value': f'{int(age)} yrs', 'impact': 35, 'level': 'Medium'})
    if stress >= 8:     factors.append({'name': 'High Stress Level',       'value': f'{stress}/10',   'impact': 55, 'level': 'Medium'})
    if sleep < 6:       factors.append({'name': 'Sleep Deprivation',       'value': f'{sleep} hrs',   'impact': 45, 'level': 'Medium'})
    if data.get('alcohol_consumption') == 'Heavy':
        factors.append({'name': 'Heavy Alcohol Use',    'value': 'Heavy',     'impact': 60, 'level': 'High'})
    if data.get('physical_activity') == 'Sedentary':
        factors.append({'name': 'Physical Inactivity',  'value': 'Sedentary', 'impact': 50, 'level': 'Medium'})
    if data.get('family_history_heart'):
        factors.append({'name': 'Family History — Heart',    'value': 'Positive', 'impact': 62, 'level': 'High'})
    if data.get('family_history_diabetes'):
        factors.append({'name': 'Family History — Diabetes', 'value': 'Positive', 'impact': 55, 'level': 'Medium'})
    factors.sort(key=lambda x: x['impact'], reverse=True)
    return factors[:8]

def generate_recommendations(data, predictions):
    recs   = []
    bmi    = float(data.get('bmi', 25))
    bp_sys = float(data.get('blood_pressure_systolic', 120))
    chol   = float(data.get('cholesterol', 180))
    stress = float(data.get('stress_level', 5))
    sleep  = float(data.get('sleep_hours', 7))

    if predictions.get('diabetes',      {}).get('probability', 0) > 40:
        recs.append({'category': 'Diabetes Prevention', 'priority': 'High',
            'text': 'Monitor blood glucose regularly. Reduce refined sugar and carb intake. Schedule HbA1c test every 3 months.'})
    if predictions.get('heart_disease', {}).get('probability', 0) > 40:
        recs.append({'category': 'Cardiac Health', 'priority': 'High',
            'text': 'Schedule ECG and lipid profile. Begin supervised cardiac exercise program. Reduce saturated fat intake.'})
    if bp_sys > 130:
        recs.append({'category': 'Blood Pressure', 'priority': 'High',
            'text': 'Reduce sodium to <2g/day. Practice daily meditation. Monitor BP twice daily and consult a cardiologist.'})
    if bmi > 27:
        recs.append({'category': 'Weight Management', 'priority': 'Medium',
            'text': 'Target 5-7% weight reduction. Combine 150 min/week aerobic exercise with a 500 kcal/day deficit.'})
    if data.get('smoking'):
        recs.append({'category': 'Smoking Cessation', 'priority': 'High',
            'text': 'Enroll in a cessation program. Nicotine replacement therapy with counselling improves success rates significantly.'})
    if chol > 200:
        recs.append({'category': 'Cholesterol', 'priority': 'Medium',
            'text': 'Adopt a Mediterranean diet rich in omega-3. Increase soluble fibre. Re-check lipid panel in 6 weeks.'})
    if stress > 7:
        recs.append({'category': 'Stress Management', 'priority': 'Medium',
            'text': 'Practice mindfulness-based stress reduction (MBSR). Target 7-9 hours sleep. Consider cognitive behavioural therapy.'})
    if sleep < 6:
        recs.append({'category': 'Sleep Hygiene', 'priority': 'Medium',
            'text': 'Set consistent sleep/wake times. Avoid screens 1 hr before bed. Evaluate for sleep apnea if snoring is present.'})
    if predictions.get('obesity',       {}).get('probability', 0) > 50:
        recs.append({'category': 'Obesity Risk', 'priority': 'High',
            'text': 'Consult a nutritionist for a structured meal plan. Aim for 10,000 steps/day and progressive resistance training.'})
    if not recs:
        recs.append({'category': 'General Wellness', 'priority': 'Low',
            'text': 'Maintain your healthy lifestyle. Annual full health screening recommended. Keep up regular activity and balanced nutrition.'})
    return recs

def generate_alerts(predictions, overall):
    alerts = []
    for disease, info in predictions.items():
        label = disease.replace('_', ' ').title()
        if info['level'] == 'Critical':
            alerts.append({'type': label, 'severity': 'Critical',
                'message': f"CRITICAL: {label} risk is {info['probability']}%. Immediate medical consultation required."})
        elif info['level'] == 'High':
            alerts.append({'type': label, 'severity': 'Warning',
                'message': f"HIGH RISK: {label} at {info['probability']}%. Schedule a physician visit within 2 weeks."})
    if overall > 60:
        alerts.append({'type': 'Overall Risk', 'severity': 'Critical',
            'message': f'Overall health risk score {overall}% — multi-specialist consultation strongly advised.'})
    return alerts

# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return response

@app.route('/api/<path:_>', methods=['OPTIONS'])
def options_handler(_):
    return '', 204

# ─────────────────────────────────────────────────────────────
# STATIC / FRONTEND
# ─────────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, 'index.html')

# ─────────────────────────────────────────────────────────────
# API — HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health_check():
    db_ok = get_db() is not None
    return jsonify({
        'status':            'ok',
        'version':           '3.0',
        'db_connected':      db_ok,
        'models_loaded':     list(MODELS.keys()),
        'total_patients':    len(patients_db),
        'total_assessments': len(assessments_db),
        'timestamp':         datetime.now().isoformat(),
    })

# ─────────────────────────────────────────────────────────────
# API — PATIENTS
# ─────────────────────────────────────────────────────────────
@app.route('/api/patients', methods=['GET'])
def get_patients():
    conn = get_db()
    if not conn:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT
                p.*,
                rp.overall_risk_score,
                rp.risk_category,
                rp.prediction_date       AS latest_assessment_date,
                COUNT(rp2.id)            AS assessment_count
            FROM patients p
            LEFT JOIN risk_predictions rp
                ON p.patient_id = rp.patient_id
                AND rp.id = (
                    SELECT MAX(id) FROM risk_predictions
                    WHERE patient_id = p.patient_id
                )
            LEFT JOIN risk_predictions rp2
                ON p.patient_id = rp2.patient_id
            GROUP BY p.patient_id
            ORDER BY p.created_at DESC
        """)
        rows = cur.fetchall()
        patients = [{k: (v.isoformat() if hasattr(v, 'isoformat') else v)
                     for k, v in row.items()} for row in rows]
        return jsonify({'success': True, 'data': patients, 'total': len(patients)})
    except Error as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route('/api/patients', methods=['POST'])
def create_patient():
    data = request.get_json() or {}
    if not data.get('name'):
        return jsonify({'success': False, 'message': 'Patient name is required'}), 400

    conn = get_db()
    if not conn:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(dictionary=True)

        # Auto-generate next patient ID
        cur.execute("SELECT patient_id FROM patients ORDER BY patient_id DESC LIMIT 1")
        last = cur.fetchone()
        new_pid = f"P{str(int(last['patient_id'][1:]) + 1).zfill(3)}" if last else 'P001'

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute("""
            INSERT INTO patients (patient_id, name, age, gender, email, phone, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (new_pid, data['name'], int(data.get('age', 0)),
              data.get('gender', 'Unknown'), data.get('email', ''),
              data.get('phone', ''), now))
        conn.commit()

        patient = {'patient_id': new_pid, 'name': data['name'],
                   'age': int(data.get('age', 0)), 'gender': data.get('gender', 'Unknown'),
                   'email': data.get('email', ''), 'phone': data.get('phone', ''),
                   'created_at': now}
        patients_db[new_pid] = patient   # update cache

        return jsonify({'success': True, 'data': patient,
                        'message': f'Patient {new_pid} registered successfully'})
    except Error as e:
        conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    conn = get_db()
    if not conn:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
        patient = cur.fetchone()
        if not patient:
            return jsonify({'success': False, 'message': 'Patient not found'}), 404

        cur.execute("SELECT * FROM risk_predictions WHERE patient_id = %s ORDER BY prediction_date DESC", (patient_id,))
        assessments = cur.fetchall()

        def serialize(obj):
            return {k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k, v in obj.items()}

        return jsonify({'success': True, 'data': {
            **serialize(patient),
            'assessments':      [serialize(a) for a in assessments],
            'assessment_count': len(assessments),
        }})
    except Error as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route('/api/patients/<patient_id>/history', methods=['GET'])
def patient_history(patient_id):
    conn = get_db()
    if not conn:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM risk_predictions WHERE patient_id = %s ORDER BY prediction_date ASC", (patient_id,))
        rows = cur.fetchall()
        history = [{k: (v.isoformat() if hasattr(v, 'isoformat') else v)
                    for k, v in row.items()} for row in rows]
        return jsonify({'success': True, 'data': history, 'total': len(history)})
    except Error as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()

# ─────────────────────────────────────────────────────────────
# API — CORE ML ASSESSMENT  ← saves to MySQL
# ─────────────────────────────────────────────────────────────
@app.route('/api/assess', methods=['POST'])
def assess_patient():
    data       = request.get_json() or {}
    patient_id = data.get('patient_id')

    # Validate patient exists
    if patient_id:
        conn_chk = get_db()
        if conn_chk:
            c = conn_chk.cursor(dictionary=True)
            c.execute("SELECT patient_id FROM patients WHERE patient_id = %s", (patient_id,))
            if not c.fetchone():
                c.close(); conn_chk.close()
                return jsonify({'success': False, 'message': f'Patient {patient_id} not found'}), 404
            c.close(); conn_chk.close()

    if not MODELS:
        return jsonify({'success': False, 'message': 'ML models not loaded'}), 503

    try:
        features     = build_feature_vector(data)
        predictions  = predict_risks(features)
        overall      = calculate_overall_score(predictions)
        category     = risk_category(overall)
        risk_factors = generate_risk_factors(data, predictions)
        recs         = generate_recommendations(data, predictions)
        alerts       = generate_alerts(predictions, overall)
        now          = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # ── Save to MySQL ──────────────────────────────────────
        db_id = None
        conn  = get_db()
        if conn:
            try:
                cur = conn.cursor()

                # 1. health_records
                cur.execute("""
                    INSERT INTO health_records (
                        patient_id, record_date,
                        blood_pressure_systolic, blood_pressure_diastolic,
                        heart_rate, bmi, blood_glucose, cholesterol,
                        smoking_status, alcohol_consumption, physical_activity,
                        diet_quality, sleep_hours, stress_level,
                        family_history_diabetes, family_history_heart_disease,
                        family_history_hypertension, family_history_cancer
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    patient_id or 'GUEST', now,
                    data.get('blood_pressure_systolic'),
                    data.get('blood_pressure_diastolic'),
                    data.get('heart_rate'),
                    data.get('bmi'),
                    data.get('blood_glucose'),
                    data.get('cholesterol'),
                    '1' if data.get('smoking') else '0',
                    data.get('alcohol_consumption', 'None'),
                    data.get('physical_activity', 'Sedentary'),
                    data.get('diet_quality', 'Fair'),
                    data.get('sleep_hours'),
                    data.get('stress_level'),
                    int(bool(data.get('family_history_diabetes', 0))),
                    int(bool(data.get('family_history_heart', 0))),
                    int(bool(data.get('family_history_hypertension', 0))),
                    int(bool(data.get('family_history_cancer', 0))),
                ))
                record_id = cur.lastrowid

                # 2. risk_predictions
                cur.execute("""
                    INSERT INTO risk_predictions (
                        patient_id, record_id,
                        diabetes_risk, heart_disease_risk,
                        hypertension_risk, stroke_risk, obesity_risk,
                        overall_risk_score, risk_category,
                        recommendations, prediction_date
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    patient_id or 'GUEST', record_id,
                    predictions['diabetes']['probability'],
                    predictions['heart_disease']['probability'],
                    predictions['hypertension']['probability'],
                    predictions['stroke']['probability'],
                    predictions['obesity']['probability'],
                    overall, category,
                    json.dumps([r['text'] for r in recs]),
                    now,
                ))
                db_id = cur.lastrowid

                # 3. risk_factors
                for f in risk_factors:
                    cur.execute("""
                        INSERT INTO risk_factors
                            (prediction_id, factor_name, factor_value, impact_level, contribution_percentage)
                        VALUES (%s,%s,%s,%s,%s)
                    """, (db_id, f['name'], f['value'], f['level'], f['impact']))

                conn.commit()
                print(f"  [DB] Saved — patient={patient_id or 'GUEST'}, assessment_id={db_id}")

            except Error as e:
                conn.rollback()
                print(f"  [DB WARN] Save failed: {e}")
            finally:
                cur.close()
                conn.close()
        # ──────────────────────────────────────────────────────

        assessment = {
            'id':                       db_id or assessment_counter[0],
            'patient_id':               patient_id or 'GUEST',
            'age':                      data.get('age'),
            'gender':                   data.get('gender'),
            'bmi':                      data.get('bmi'),
            'blood_pressure_systolic':  data.get('blood_pressure_systolic'),
            'blood_pressure_diastolic': data.get('blood_pressure_diastolic'),
            'blood_glucose':            data.get('blood_glucose'),
            'cholesterol':              data.get('cholesterol'),
            'heart_rate':               data.get('heart_rate'),
            'smoking':                  data.get('smoking'),
            'alcohol_consumption':      data.get('alcohol_consumption'),
            'physical_activity':        data.get('physical_activity'),
            'diet_quality':             data.get('diet_quality'),
            'sleep_hours':              data.get('sleep_hours'),
            'stress_level':             data.get('stress_level'),
            'risk_predictions':         predictions,
            'overall_risk_score':       overall,
            'risk_category':            category,
            'risk_factors':             risk_factors,
            'recommendations':          recs,
            'alerts':                   alerts,
            'assessed_at':              now,
        }

        assessments_db.append(assessment)
        assessment_counter[0] += 1

        return jsonify({'success': True, 'data': {
            'assessment':         assessment,
            'predictions':        predictions,
            'overall_risk_score': overall,
            'risk_category':      category,
            'risk_factors':       risk_factors,
            'recommendations':    recs,
            'alerts':             alerts,
        }})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction error: {str(e)}'}), 500

# ─────────────────────────────────────────────────────────────
# API — DASHBOARD STATS
# ─────────────────────────────────────────────────────────────
@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    conn = get_db()
    if not conn:
        # Fallback to in-memory
        risk_dist = {'Low': 0, 'Moderate': 0, 'High': 0, 'Critical': 0}
        for a in assessments_db:
            risk_dist[a.get('risk_category', 'Low')] += 1
        avg = round(sum(a['overall_risk_score'] for a in assessments_db) / max(len(assessments_db), 1), 1)
        return jsonify({'success': True, 'data': {
            'total_patients': len(patients_db), 'total_assessments': len(assessments_db),
            'risk_distribution': risk_dist, 'average_risk_score': avg,
            'disease_rates': {}, 'recent_assessments': [],
        }})
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT COUNT(*) AS cnt FROM patients")
        total_patients = cur.fetchone()['cnt']

        cur.execute("SELECT COUNT(*) AS cnt FROM risk_predictions")
        total_assessments = cur.fetchone()['cnt']

        cur.execute("SELECT risk_category, COUNT(*) AS cnt FROM risk_predictions GROUP BY risk_category")
        risk_dist = {'Low': 0, 'Moderate': 0, 'High': 0, 'Critical': 0}
        for row in cur.fetchall():
            risk_dist[row['risk_category']] = row['cnt']

        cur.execute("SELECT AVG(overall_risk_score) AS avg FROM risk_predictions")
        avg_row = cur.fetchone()['avg']
        avg_score = round(float(avg_row), 1) if avg_row else 0.0

        cur.execute("""
            SELECT AVG(diabetes_risk) AS diabetes, AVG(heart_disease_risk) AS heart_disease,
                   AVG(hypertension_risk) AS hypertension, AVG(stroke_risk) AS stroke,
                   AVG(obesity_risk) AS obesity
            FROM risk_predictions
        """)
        rates_row = cur.fetchone()
        disease_rates = {k: round(float(v), 1) if v else 0.0 for k, v in rates_row.items()}

        cur.execute("""
            SELECT rp.*, p.name AS patient_name
            FROM risk_predictions rp
            LEFT JOIN patients p ON rp.patient_id = p.patient_id
            ORDER BY rp.prediction_date DESC LIMIT 5
        """)
        recent = [{k: (v.isoformat() if hasattr(v, 'isoformat') else v)
                   for k, v in row.items()} for row in cur.fetchall()]

        return jsonify({'success': True, 'data': {
            'total_patients':     total_patients,
            'total_assessments':  total_assessments,
            'risk_distribution':  risk_dist,
            'average_risk_score': avg_score,
            'disease_rates':      disease_rates,
            'recent_assessments': recent,
        }})
    except Error as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()

# ─────────────────────────────────────────────────────────────
# API — POPULATION ANALYTICS
# ─────────────────────────────────────────────────────────────
@app.route('/api/analytics/population', methods=['GET'])
def population_analytics():
    conn = get_db()
    if not conn:
        return jsonify({'success': True, 'data': {'by_age': [], 'by_gender': []}})
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT
                CASE
                    WHEN p.age < 30 THEN '18-29'
                    WHEN p.age < 45 THEN '30-44'
                    WHEN p.age < 60 THEN '45-59'
                    ELSE '60+'
                END AS age_group,
                AVG(rp.overall_risk_score) AS avg_risk,
                COUNT(*) AS cnt
            FROM risk_predictions rp
            JOIN patients p ON rp.patient_id = p.patient_id
            GROUP BY age_group
        """)
        by_age = [{'age_group': r['age_group'],
                   'avg_risk':  round(float(r['avg_risk']), 1),
                   'count':     r['cnt']} for r in cur.fetchall()]

        cur.execute("""
            SELECT p.gender,
                   AVG(rp.overall_risk_score) AS avg_risk,
                   COUNT(*) AS cnt
            FROM risk_predictions rp
            JOIN patients p ON rp.patient_id = p.patient_id
            GROUP BY p.gender
        """)
        by_gender = [{'gender':   r['gender'],
                      'avg_risk': round(float(r['avg_risk']), 1),
                      'count':    r['cnt']} for r in cur.fetchall()]

        return jsonify({'success': True, 'data': {'by_age': by_age, 'by_gender': by_gender}})
    except Error as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
# ── Startup — runs under gunicorn AND python directly ──────
print("=" * 55)
print("  Predictive Health Risk Analysis  v3.1")
print("=" * 55)
print("\n[1] Loading ML models...")
load_models()
print(f"    Ready: {list(MODELS.keys())}")
print("\n[2] Syncing from MySQL...")
sync_from_db()
print("\n[3] App ready.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"    Running locally at http://localhost:{port}\n")
    app.run(debug=False, host='0.0.0.0', port=port)
