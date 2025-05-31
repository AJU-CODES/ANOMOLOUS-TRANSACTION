import os

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load and preprocess dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Drop unnecessary object columns that were seen at training time
    columns_to_drop = ["Unnamed: 0", "cc_num", "zip"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

# Train models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(n_jobs=-1, max_iter=100),
        "Random Forest": RandomForestClassifier(n_jobs=-1, n_estimators=50)
    }
    
    results = {}
    best_model, best_auc = None, 0
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        results[name] = (accuracy_score(y_test, y_pred), auc_score)
        
        if auc_score > best_auc:
            best_model, best_auc = model, auc_score
    
    # ANN Model
    ann = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    ann.fit(X_train_scaled, y_train, epochs=1, batch_size=2, verbose=0)
    ann_accuracy, ann_auc = ann.evaluate(X_test_scaled, y_test, verbose=0)[1], ann.evaluate(X_test_scaled, y_test, verbose=0)[2]
    results["Artificial Neural Network"] = (ann_accuracy, ann_auc)
    
    if ann_auc > best_auc:
        best_model = ann
    
    return best_model, scaler, results, X.columns

# Save model and scaler
def save_model(model, scaler, feature_names, model_dir='model_files'):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
    print("Model, scaler, and feature names saved successfully.")

if __name__ == "__main__":
    X, y = load_and_preprocess_data(r"D:\PROJECT MCA\fraudTest.csv")
    best_model, scaler, results, feature_names = train_models(X, y)
    save_model(best_model, scaler, feature_names)
    print("Model training completed. Best model saved.")



# python train_model.py
# streamlit run app.py