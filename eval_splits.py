import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Load Data
df = pd.read_csv('student-lifestyle-and-stress-dataset.csv')

# Data Cleansing
df['Study_Hours'] = df['Study_Hours'].clip(lower=0)
df['Attendance'] = df['Attendance'].clip(0, 100)
df['Student_Type'] = df['Student_Type'].fillna(df['Student_Type'].mode()[0])

num_cols = ['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 'Attendance', 'Exam_Pressure', 'Family_Support']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df_model = pd.get_dummies(df, columns=['Student_Type'], drop_first=True)
X = df_model[['Sleep_Hours','Study_Hours','Social_Media_Hours','Attendance','Exam_Pressure']]
y = df['Stress_Level'].fillna(0)

# Test multiple test sizes
sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
print(f'Total Data: {len(df)}')
print('--- TEST SIZES ---')
for size in sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
    rf.fit(X_train, y_train)
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    print(f'Test Size {size*100:0.0f}% -> Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Gap: {train_acc - test_acc:.4f}')

# Cross Validation
print('\n--- CROSS VALIDATION (5-Fold) ---')
rf_cv = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
scores = cross_val_score(rf_cv, X, y, cv=5)
print(f'CV Scores: {scores}')
print(f'Mean CV Acc: {scores.mean():.4f} +/- {scores.std():.4f}')
