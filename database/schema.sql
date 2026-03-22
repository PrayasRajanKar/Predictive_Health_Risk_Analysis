-- ============================================================
--  Predictive Health Risk Analysis — Database Schema v3.0
--  No sample data — only real user-submitted data is stored
--  Run once:  mysql -u root -p < database/schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS health_risk_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE health_risk_db;

-- ─────────────────────────────────────────────────────────────
-- 1. PATIENTS
--    Registered by the user via the frontend form
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id          INT          AUTO_INCREMENT PRIMARY KEY,
    patient_id  VARCHAR(10)  NOT NULL UNIQUE,        -- e.g. P001, P002
    name        VARCHAR(100) NOT NULL,
    age         INT          NOT NULL,
    gender      ENUM('Male','Female','Other') NOT NULL,
    email       VARCHAR(150) DEFAULT '',
    phone       VARCHAR(30)  DEFAULT '',
    created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_patient_id (patient_id)
);

-- ─────────────────────────────────────────────────────────────
-- 2. HEALTH RECORDS
--    Raw vitals and lifestyle inputs submitted per assessment
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS health_records (
    id                              INT          AUTO_INCREMENT PRIMARY KEY,
    patient_id                      VARCHAR(10)  NOT NULL,
    record_date                     DATETIME     DEFAULT CURRENT_TIMESTAMP,

    -- Vitals
    blood_pressure_systolic         INT,
    blood_pressure_diastolic        INT,
    heart_rate                      INT,
    bmi                             DECIMAL(5,2),
    blood_glucose                   DECIMAL(6,2),
    cholesterol                     DECIMAL(6,2),

    -- Lifestyle
    smoking_status                  TINYINT(1)   DEFAULT 0,   -- 0 = No, 1 = Yes
    alcohol_consumption             ENUM('None','Occasional','Moderate','Heavy') DEFAULT 'None',
    physical_activity               ENUM('Sedentary','Light','Moderate','Active') DEFAULT 'Sedentary',
    diet_quality                    ENUM('Poor','Fair','Good','Excellent') DEFAULT 'Fair',
    sleep_hours                     DECIMAL(4,1),
    stress_level                    DECIMAL(4,1),

    -- Family history flags
    family_history_diabetes         TINYINT(1)   DEFAULT 0,
    family_history_heart_disease    TINYINT(1)   DEFAULT 0,
    family_history_hypertension     TINYINT(1)   DEFAULT 0,
    family_history_cancer           TINYINT(1)   DEFAULT 0,

    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    INDEX idx_hr_patient (patient_id),
    INDEX idx_hr_date    (record_date)
);

-- ─────────────────────────────────────────────────────────────
-- 3. RISK PREDICTIONS
--    ML output saved after every assessment
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS risk_predictions (
    id                  INT          AUTO_INCREMENT PRIMARY KEY,
    patient_id          VARCHAR(10)  NOT NULL,
    record_id           INT,                          -- FK to health_records

    -- Per-disease risk scores (0.0 – 100.0)
    diabetes_risk       DECIMAL(5,2) DEFAULT 0,
    heart_disease_risk  DECIMAL(5,2) DEFAULT 0,
    hypertension_risk   DECIMAL(5,2) DEFAULT 0,
    stroke_risk         DECIMAL(5,2) DEFAULT 0,
    obesity_risk        DECIMAL(5,2) DEFAULT 0,

    -- Aggregate
    overall_risk_score  DECIMAL(5,2) DEFAULT 0,
    risk_category       ENUM('Low','Moderate','High','Critical') DEFAULT 'Low',

    -- Serialised recommendation texts (JSON array)
    recommendations     TEXT,

    prediction_date     DATETIME     DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    FOREIGN KEY (record_id)  REFERENCES health_records(id)   ON DELETE SET NULL,
    INDEX idx_rp_patient (patient_id),
    INDEX idx_rp_date    (prediction_date),
    INDEX idx_rp_category(risk_category)
);

-- ─────────────────────────────────────────────────────────────
-- 4. RISK FACTORS
--    Individual contributing factors per prediction
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS risk_factors (
    id                      INT          AUTO_INCREMENT PRIMARY KEY,
    prediction_id           INT          NOT NULL,
    factor_name             VARCHAR(100) NOT NULL,
    factor_value            VARCHAR(100),
    impact_level            ENUM('Low','Medium','High') DEFAULT 'Low',
    contribution_percentage DECIMAL(5,2) DEFAULT 0,

    FOREIGN KEY (prediction_id) REFERENCES risk_predictions(id) ON DELETE CASCADE,
    INDEX idx_rf_prediction (prediction_id)
);

-- ─────────────────────────────────────────────────────────────
-- Verify
-- ─────────────────────────────────────────────────────────────
SELECT 'Schema created successfully.' AS status;
SHOW TABLES;