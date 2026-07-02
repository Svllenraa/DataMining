import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_export_model():
    # 1. Load Data
    df = pd.read_csv('student-lifestyle-and-stress-dataset.csv')
    
    # 2. Data Cleansing (Versi Azizah + Fix Leroy)
    df['Study_Hours'] = df['Study_Hours'].clip(lower=0)
    df['Attendance'] = df['Attendance'].clip(0, 100)
    
    num_cols = ['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 'Attendance', 'Exam_Pressure']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 3. Preprocessing untuk Model
    X = df[['Sleep_Hours','Study_Hours','Social_Media_Hours','Attendance','Exam_Pressure']]
    
    # Mengisi nilai target yang kosong dengan nilai modus
    df['Stress_Level'] = df['Stress_Level'].fillna(df['Stress_Level'].mode()[0])
    y = df['Stress_Level'] # Target
    
    # Scaling untuk Clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 'Attendance', 'Exam_Pressure']])
    
    # 4. Training (Random Forest)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 5. Training (K-Means)
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)
    
    # 6. Evaluasi
    y_test_pred = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    cm = confusion_matrix(y_test, y_test_pred)
    
    # 7. Export Aset
    assets = {
        'rf_model': rf_model,
        'kmeans_model': kmeans_model,
        'scaler': scaler,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'gap_overfitting': abs(train_acc - test_acc),
        'cm': cm
    }
    joblib.dump(assets, 'model_assets.pkl')
    print("✅ Training selesai! 'model_assets.pkl' telah diperbarui dengan data yang sudah diproses.")

if __name__ == "__main__":
    train_and_export_model()