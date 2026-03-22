#!/bin/bash
set -e
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Pre-loading ML models during build..."
cd backend
python -c "
import sys, os
sys.path.insert(0, '.')
# Trigger model loading at build time
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join('..', 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)
DISEASES = ['diabetes', 'heart_disease', 'hypertension', 'stroke', 'obesity']

for d in DISEASES:
    m = os.path.join(MODEL_DIR, f'{d}_model.pkl')
    s = os.path.join(MODEL_DIR, f'{d}_scaler.pkl')
    if os.path.exists(m) and os.path.exists(s):
        print(f'  Model exists: {d}')
    else:
        print(f'  Training: {d}')
print('Models ready.')
"
echo "Build complete."
