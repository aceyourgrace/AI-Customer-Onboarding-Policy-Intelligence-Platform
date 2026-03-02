
# -------------------------------
# train_cltv_models.py
# Train CLTV prediction models (Linear Regression, Random Forest, XGBoost)
# -------------------------------

"""
This script trains multiple regression models to predict Customer Lifetime Value (CLTV) for leads.
It includes:
1. Data loading and feature selection
2. Train/test split
3. Model training for Linear Regression, Random Forest, and XGBoost
4. Evaluation using RMSE and MAE
5. Saving trained models for later use
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Step 0: Fix imports
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.select_features import select_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Step 1: Load dataset
# -------------------------------
raw_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'bank_leads_v4.csv')
df = pd.read_csv(raw_path, sep='\t')

# -------------------------------
# Step 2: Feature selection
# -------------------------------
# We'll predict CLTV as target
X, _, numeric_features, categorical_features = select_features(df)
y = df['CLTV']  # CLTV column from original dataset

# -------------------------------
# Step 3: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Preprocessing
# -------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# -------------------------------
# Step 5: Define models
# -------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=7, random_state=42),
    "XGBRegressor": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1,
                                 objective='reg:squarederror', random_state=42)
}

# -------------------------------
# Step 6: Train each model and save pipeline
# -------------------------------
models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(models_dir, exist_ok=True)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    pipeline.fit(X_train, y_train)
    
    # Save the trained pipeline
    model_file = os.path.join(models_dir, f'cltv_{name}.pkl')
    joblib.dump(pipeline, model_file)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")
    print(f"Pipeline saved to {model_file}")

print("\nAll CLTV models trained and saved successfully!")