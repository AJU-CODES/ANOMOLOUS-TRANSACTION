import os

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import warnings
from imblearn.over_sampling import SMOTENC


warnings.filterwarnings('ignore')


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    drop_cols = ["Unnamed: 0", "cc_num", "trans_num", "trans_date_trans_time", "dob",
                 "first", "last", "street", "city", "zip"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")

    if 'is_fraud' not in df.columns:
        raise ValueError("'is_fraud' column not found.")
    print('df returned to train model function')
    return df


def reduce_cardinality(df, col, threshold=50):
    top = df[col].value_counts().nlargest(threshold).index
    df[col] = df[col].where(df[col].isin(top), 'Other')
    return df

def train_models(df):
    y = df.pop('is_fraud')
    
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Reduce cardinality
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df = reduce_cardinality(df, col, threshold=50)

    # Get categorical indices
    cat_indices = [df.columns.get_loc(col) for col in categorical_cols]

    # SMOTENC with reduced memory
    smote = SMOTENC(categorical_features=cat_indices, random_state=42, k_neighbors=2)
    X_resampled, y_resampled = smote.fit_resample(df, y)

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    X_encoded = preprocessor.fit_transform(X_resampled)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_resampled, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_jobs=-1, n_estimators=50)
    }

    results = {}
    best_model, best_auc = None, 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results[name] = (accuracy_score(y_test, model.predict(X_test)), auc_score)
        if auc_score > best_auc:
            best_model, best_auc = model, auc_score

    ann = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    ann.fit(X_train, y_train, epochs=1, batch_size=2, verbose=0)
    ann_accuracy, ann_auc = ann.evaluate(X_test, y_test, verbose=0)[1], ann.evaluate(X_test, y_test, verbose=0)[2]
    results["ANN"] = (ann_accuracy, ann_auc)

    if ann_auc > best_auc:
        best_model = ann

    return best_model, preprocessor, results


# Save model and scaler
def save_model(model,preprocessor , model_dir='model_files'):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    print("Model and preprocessor saved successfully.")

if __name__ == "__main__":
    df = load_and_preprocess_data(r"D:\PROJECT MCA\fraudTest.csv")
    best_model,preprocessor,results = train_models(df)
    save_model(best_model, preprocessor)
    print("Model training completed. Best model saved.")
    print("Model training completed. Best model saved.\nResults:")
    for name, (acc, auc) in results.items():
        print(f"{name}: Accuracy = {acc:.4f}, AUC = {auc:.4f}")

