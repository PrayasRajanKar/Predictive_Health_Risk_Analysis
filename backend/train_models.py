"""
Train and save ML models for health risk prediction.
Run this once to generate model files.
"""
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

np.random.seed(42)
N = 5000

def generate_dataset():
    age = np.random.randint(18, 85, N)
    bmi = np.random.normal(26, 5, N).clip(15, 50)
    bp_sys = np.random.normal(120, 20, N).clip(80, 200)
    bp_dia = np.random.normal(80, 12, N).clip(50, 130)
    glucose = np.random.normal(95, 25, N).clip(60, 300)
    cholesterol = np.random.normal(195, 40, N).clip(100, 350)
    heart_rate = np.random.normal(75, 12, N).clip(40, 140)
    smoking = np.random.randint(0, 2, N)
    alcohol = np.random.randint(0, 4, N)
    physical_activity = np.random.randint(0, 4, N)
    diet_quality = np.random.randint(0, 4, N)
    sleep_hours = np.random.normal(7, 1.5, N).clip(3, 12)
    stress_level = np.random.randint(1, 11, N)
    fam_diabetes = np.random.randint(0, 2, N)
    fam_heart = np.random.randint(0, 2, N)
    fam_hypertension = np.random.randint(0, 2, N)
    fam_cancer = np.random.randint(0, 2, N)

    X = np.column_stack([
        age, bmi, bp_sys, bp_dia, glucose, cholesterol, heart_rate,
        smoking, alcohol, physical_activity, diet_quality,
        sleep_hours, stress_level,
        fam_diabetes, fam_heart, fam_hypertension, fam_cancer
    ])

    # Diabetes risk
    diabetes_score = (
        (glucose > 126) * 0.4 + (bmi > 30) * 0.2 + (age > 45) * 0.15 +
        fam_diabetes * 0.15 + (physical_activity < 2) * 0.1
    )
    y_diabetes = (diabetes_score + np.random.normal(0, 0.1, N) > 0.35).astype(int)

    # Heart disease risk
    heart_score = (
        (bp_sys > 140) * 0.3 + (cholesterol > 240) * 0.25 + (age > 55) * 0.15 +
        smoking * 0.15 + fam_heart * 0.15
    )
    y_heart = (heart_score + np.random.normal(0, 0.1, N) > 0.35).astype(int)

    # Hypertension risk
    hyp_score = (
        (bp_sys > 130) * 0.35 + (bmi > 28) * 0.2 + (stress_level > 7) * 0.15 +
        fam_hypertension * 0.15 + (alcohol > 2) * 0.15
    )
    y_hypertension = (hyp_score + np.random.normal(0, 0.1, N) > 0.35).astype(int)

    # Stroke risk
    stroke_score = (
        (bp_sys > 150) * 0.35 + (age > 65) * 0.25 + smoking * 0.2 +
        (cholesterol > 250) * 0.1 + fam_heart * 0.1
    )
    y_stroke = (stroke_score + np.random.normal(0, 0.1, N) > 0.3).astype(int)

    # Obesity risk
    obesity_score = (
        (bmi > 30) * 0.5 + (physical_activity < 1) * 0.2 +
        (diet_quality < 1) * 0.2 + (sleep_hours < 6) * 0.1
    )
    y_obesity = (obesity_score + np.random.normal(0, 0.1, N) > 0.3).astype(int)

    return X, y_diabetes, y_heart, y_hypertension, y_stroke, y_obesity


def build_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])


def train_all():
    print("Generating synthetic training data...")
    X, y_dia, y_heart, y_hyp, y_stroke, y_obs = generate_dataset()

    os.makedirs('ml_models', exist_ok=True)

    targets = {
        'diabetes': y_dia,
        'heart_disease': y_heart,
        'hypertension': y_hyp,
        'stroke': y_stroke,
        'obesity': y_obs
    }

    models_config = {
        'diabetes': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4),
        'heart_disease': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4),
        'hypertension': RandomForestClassifier(n_estimators=200, max_depth=6),
        'stroke': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3),
        'obesity': RandomForestClassifier(n_estimators=150, max_depth=5)
    }

    for name, y in targets.items():
        print(f"Training {name} model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline = build_pipeline(models_config[name])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        print(f"  Accuracy: {score:.3f}")
        joblib.dump(pipeline, f'ml_models/{name}_model.pkl')
        print(f"  Saved ml_models/{name}_model.pkl")

    feature_names = [
        'age', 'bmi', 'bp_systolic', 'bp_diastolic', 'blood_glucose',
        'cholesterol', 'heart_rate', 'smoking', 'alcohol', 'physical_activity',
        'diet_quality', 'sleep_hours', 'stress_level',
        'family_history_diabetes', 'family_history_heart',
        'family_history_hypertension', 'family_history_cancer'
    ]
    joblib.dump(feature_names, 'ml_models/feature_names.pkl')
    print("\nAll models trained and saved successfully!")


if __name__ == '__main__':
    train_all()