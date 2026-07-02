import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================================
# 1. LOAD DATA MENTAH
# =====================================================================
print("⏳ Membaca dataset...")
try:
    df_raw = pd.read_csv('student-lifestyle-and-stress-dataset.csv')
    print("✅ Dataset berhasil dimuat!")
except FileNotFoundError:
    print("❌ File 'student-lifestyle-and-stress-dataset.csv' tidak ditemukan!")
    print("Pastikan file ini berada di folder yang sama dengan script generate_grafik.py")
    exit()

# Mengatur tema visual grafik agar rapi dan estetik
sns.set_theme(style="whitegrid")

print("\n🎨 Sedang memproses data dan menghasilkan grafik... Mohon tunggu pop-up muncul.")
print("💡 TIPS: Setelah meng-screenshot grafik yang muncul, TUTUP (klik X) jendela grafik tersebut agar grafik berikutnya bisa muncul.")

# =====================================================================
# 2. VISUALISASI DATA AWAL (EKSPLORASI DATA)
# =====================================================================

# Grafik 1: Distribusi Jenis Pelajar (Student Type)
plt.figure(figsize=(8, 5))
sns.countplot(data=df_raw, x='Student_Type', palette='Set2')
plt.title('Distribusi Jenis Pelajar dalam Dataset', fontsize=12, fontweight='bold')
plt.xlabel('Tipe Pelajar')
plt.ylabel('Jumlah Responden')
plt.tight_layout()
plt.show()

# Grafik 2: Hubungan Jam Tidur Terhadap Tingkat Stres
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_raw, x='Stress_Level', y='Sleep_Hours', hue='Stress_Level', palette='coolwarm', legend=False)
plt.title('Hubungan Alokasi Jam Tidur dengan Tingkat Stres', fontsize=12, fontweight='bold')
plt.xlabel('Tingkat Stres (0 = Aman, 1 = Stres Tinggi)')
plt.ylabel('Jam Tidur Harian (Sleep Hours)')
plt.tight_layout()
plt.show()


# =====================================================================
# 3. PREPROCESSING (PETA DATA KOSONG SEBELUM DI-CLEANSING)
# =====================================================================

# Grafik 3: Peta Sebaran Missing Values
plt.figure(figsize=(10, 5))
sns.heatmap(df_raw.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Peta Sebaran Missing Values Sebelum Prapemrosesan', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()


# =====================================================================
# 4. GABUNGAN LOGIKA CLEANSING (VERSI AZIZAH + FIX LEROY)
# =====================================================================
df_clean = df_raw.copy()

# 1. Data Clipping
df_clean['Study_Hours'] = df_clean['Study_Hours'].clip(lower=0)
df_clean['Attendance'] = df_clean['Attendance'].clip(0, 100)

# 2. Imputasi Mode untuk Kategorikal (meskipun fitur tidak digunakan ke RF)
df_clean['Student_Type'] = df_clean['Student_Type'].fillna(df_clean['Student_Type'].mode()[0])

# 3. Imputasi Median untuk Numerik
num_cols = ['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 'Attendance', 'Exam_Pressure']
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("✅ Tahap Pra-pemrosesan Data (Data Cleansing) selesai!")


# =====================================================================
# 5. K-MEANS CLUSTERING (METODE ELBOW, TABEL WCSS & SCATTER PLOT)
# =====================================================================
print("\n=== MEMPROSES K-MEANS CLUSTERING ===")

# Scaling data bersih menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 'Attendance', 'Exam_Pressure']])
print("Data berhasil discaling!")

# Perhitungan WCSS untuk Tabel & Grafik Elbow
wcss = []
k_range = range(1, 11)
for i in k_range:
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Menampilkan TABEL PERBANDINGAN WCSS
tabel_wcss = pd.DataFrame({'Jumlah Klaster (K)': list(k_range), 'Nilai WCSS': wcss})
print("\n📊 TABEL PERBANDINGAN WCSS:")
print(tabel_wcss.to_string(index=False))

# Grafik 4: Metode Elbow
plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b', linewidth=2)
plt.axvline(x=3, color='r', linestyle=':', label='Titik Elbow Optimal (K=3)')
plt.title('Metode Elbow untuk Penentuan Jumlah Klaster Optimal', fontsize=12, fontweight='bold')
plt.xlabel('Jumlah Klaster (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.legend()
plt.tight_layout()
plt.show() 

# Fit K-Means Final (K=3) dan tampilkan Scatter Plot
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Grafik 5: Sebaran Klaster Final (K=3)
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df_clean, x='Sleep_Hours', y='Social_Media_Hours', hue='Cluster', palette='Set1', alpha=0.6)
plt.title('Sebaran Klaster Berdasarkan Pola Tidur dan Media Sosial', fontsize=12, fontweight='bold')
plt.xlabel('Jam Tidur (Sleep Hours)')
plt.ylabel('Jam Media Sosial (Social Media Hours)')
plt.legend(title='Klaster Persona')
plt.tight_layout()
plt.show()


# =====================================================================
# 6. RANDOM FOREST CLASSIFIER (PELATIHAN, PENGUJIAN & CONFUSION MATRIX)
# =====================================================================
print("\n=== MEMPROSES RANDOM FOREST CLASSIFIER ===")

# Tentukan Fitur dan Target (Stress_Level) - diselaraskan dengan update train_model.py
X_rf = df_clean[['Sleep_Hours','Study_Hours','Social_Media_Hours','Attendance','Exam_Pressure']]
y_rf = df_clean['Stress_Level'].fillna(df_clean['Stress_Level'].mode()[0]) 

# Split Data 80% Train : 20% Test
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Latih Model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Menampilkan AKURASI DATA PELATIHAN & DATA PENGUJIAN
print(f"🎯 Akurasi Data Pelatihan: {accuracy_score(y_train, rf_model.predict(X_train)) * 100:.2f}%")
print(f"🎯 Akurasi Pengujian Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Menampilkan RECALL & LAPORAN KLASIFIKASI LENGKAP
print("\n📋 LAPORAN KLASIFIKASI LENGKAP (Perhatikan kolom 'recall'):")
print(classification_report(y_test, y_pred))

# Grafik 6: Menampilkan VISUALISASI CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Random Forest', fontsize=12, fontweight='bold')
plt.ylabel('Kategori Aktual / Sebenarnya')
plt.xlabel('Kategori Prediksi Model')
plt.tight_layout()
plt.show()
