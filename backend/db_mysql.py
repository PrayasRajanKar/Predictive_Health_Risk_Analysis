"""
MySQL Database Connector Module
Replace the in-memory store in app.py with this for production use.

Usage:
  from db_mysql import get_db, close_db, init_db

  # In your Flask app:
  app.teardown_appcontext(close_db)
  init_db(app)
"""
import mysql.connector
from mysql.connector import pooling
from flask import g
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'health_risk_db'),
    'charset': 'utf8mb4',
    'autocommit': True,
}

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name='health_risk_pool',
            pool_size=5,
            **DB_CONFIG
        )
    return _pool


def get_db():
    """Get a database connection from the pool."""
    if 'db' not in g:
        g.db = get_pool().get_connection()
    return g.db


def close_db(e=None):
    """Return connection to pool."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def execute_query(query, params=None, fetch=True):
    """Execute a query and optionally fetch results."""
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(query, params or ())
    if fetch:
        result = cursor.fetchall()
        cursor.close()
        return result
    cursor.close()
    return None


def execute_insert(query, params=None):
    """Execute an INSERT and return the last row id."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(query, params or ())
    last_id = cursor.lastrowid
    cursor.close()
    return last_id


def init_db(app):
    """Initialize database, create tables if not exist."""
    app.teardown_appcontext(close_db)
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'schema.sql')
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        with open(schema_path, 'r') as f:
            statements = f.read().split(';')
            for stmt in statements:
                stmt = stmt.strip()
                if stmt:
                    try:
                        cursor.execute(stmt)
                    except mysql.connector.Error as e:
                        if e.errno not in (1050, 1062):  # Table exists, duplicate entry
                            print(f"DB init warning: {e}")
        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized successfully.")
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        print("Running in demo mode with in-memory storage.")


# ─── Patient CRUD ─────────────────────────────────────────────────────────────

def db_get_patients():
    return execute_query("""
        SELECT p.*, 
               COUNT(a.id) as assessment_count,
               MAX(a.overall_risk_score) as latest_risk_score,
               (SELECT risk_category FROM health_assessments 
                WHERE patient_id = p.patient_id 
                ORDER BY assessed_at DESC LIMIT 1) as latest_risk_category
        FROM patients p
        LEFT JOIN health_assessments a ON p.patient_id = a.patient_id
        GROUP BY p.patient_id
        ORDER BY p.created_at DESC
    """)


def db_create_patient(data):
    from datetime import datetime
    pid_row = execute_query("SELECT COUNT(*) as cnt FROM patients")
    new_num = (pid_row[0]['cnt'] if pid_row else 0) + 1
    patient_id = f"P{str(new_num).zfill(3)}"
    execute_insert("""
        INSERT INTO patients (patient_id, name, age, gender, email, phone)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (patient_id, data['name'], data['age'], data['gender'],
          data.get('email', ''), data.get('phone', '')))
    return patient_id


def db_save_assessment(patient_id, assessment_data, predictions, overall_score, risk_category, recs, alerts):
    import json
    assessment_id = execute_insert("""
        INSERT INTO health_assessments 
        (patient_id, bmi, blood_pressure_systolic, blood_pressure_diastolic,
         blood_glucose, cholesterol, heart_rate, smoking, alcohol_consumption,
         physical_activity, diet_quality, sleep_hours, stress_level,
         family_history_diabetes, family_history_heart, family_history_hypertension,
         family_history_cancer, risk_predictions, overall_risk_score, risk_category)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        patient_id,
        assessment_data.get('bmi'), assessment_data.get('blood_pressure_systolic'),
        assessment_data.get('blood_pressure_diastolic'), assessment_data.get('blood_glucose'),
        assessment_data.get('cholesterol'), assessment_data.get('heart_rate'),
        assessment_data.get('smoking', 0), assessment_data.get('alcohol_consumption', 'None'),
        assessment_data.get('physical_activity', 'Sedentary'), assessment_data.get('diet_quality', 'Fair'),
        assessment_data.get('sleep_hours', 7), assessment_data.get('stress_level', 5),
        assessment_data.get('family_history_diabetes', 0), assessment_data.get('family_history_heart', 0),
        assessment_data.get('family_history_hypertension', 0), assessment_data.get('family_history_cancer', 0),
        json.dumps(predictions), overall_score, risk_category
    ))

    # Save recommendations
    for rec in recs:
        execute_insert("""
            INSERT INTO recommendations (assessment_id, category, recommendation, priority)
            VALUES (%s, %s, %s, %s)
        """, (assessment_id, rec['category'], rec['recommendation'], rec['priority']))

    # Save alerts
    for alert in alerts:
        execute_insert("""
            INSERT INTO risk_alerts (patient_id, assessment_id, alert_type, severity, message)
            VALUES (%s, %s, %s, %s, %s)
        """, (patient_id, assessment_id, alert['type'], alert['severity'], alert['message']))

    return assessment_id


def db_get_dashboard_stats():
    total_patients = execute_query("SELECT COUNT(*) as cnt FROM patients")[0]['cnt']
    total_assessments = execute_query("SELECT COUNT(*) as cnt FROM health_assessments")[0]['cnt']

    risk_dist_rows = execute_query("""
        SELECT risk_category, COUNT(*) as cnt 
        FROM health_assessments 
        GROUP BY risk_category
    """)
    risk_dist = {r['risk_category']: r['cnt'] for r in risk_dist_rows}

    avg_score = execute_query("SELECT AVG(overall_risk_score) as avg FROM health_assessments")[0]['avg'] or 0

    recent = execute_query("""
        SELECT a.*, p.name as patient_name
        FROM health_assessments a
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        ORDER BY a.assessed_at DESC LIMIT 5
    """)

    return {
        'total_patients': total_patients,
        'total_assessments': total_assessments,
        'risk_distribution': risk_dist,
        'average_risk_score': round(float(avg_score), 1),
        'recent_assessments': recent,
    }