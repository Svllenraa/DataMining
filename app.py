import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Ruang Tengah",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CSS STARTUP DASHBOARD STYLE
# ==================================================
st.markdown("""
<style>

[data-testid="collapsedControl"] {
    display: none;
}

/* Background utama */
.stApp {
    background-color: #F5F7F6;
}

/* Container */
.block-container {
    padding-top: 4rem;
    padding-left: 4rem;
    padding-right: 4rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #DFE7E5;
    border-right: none;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* Typography */
h1 {
    color: #2F3E46;
    font-weight: 800;
}

h2, h3 {
    color: #2F3E46;
}

p, label, li {
    color: #52616B;
}

/* Card */
.custom-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    border: 1px solid #E5ECE9;
}

/* Hero */
.hero {
    background: linear-gradient(
        135deg,
        #84A98C,
        #52796F
    );

    padding: 35px;
    border-radius: 25px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    margin-bottom: 5px;
}

.hero p {
    color: rgba(255,255,255,0.9);
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background-color: #84A98C;
    color: white;
    font-weight: 600;
    padding: 0.75rem;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #52796F;
    transform: translateY(-2px);
}

/* Metric */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E5ECE9;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
}

/* Progress */
.stProgress > div > div > div {
    background-color: #84A98C;
}

.question-card {
    background: white;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    border: 1px solid #E5ECE9;
    box-shadow: 0 6px 16px rgba(0,0,0,0.04);
}

.question-title {
    color: #2F3E46;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 15px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 20px;
    padding: 15px;
    border: 1px solid #E5ECE9;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_assets():
    try:
        return joblib.load("model_assets.pkl")
    except FileNotFoundError:
        return None

assets = load_assets()

if assets is None:
    st.error(
        "File model_assets.pkl tidak ditemukan.\n\n"
        "Pastikan model sudah dilatih terlebih dahulu."
    )
    st.stop()

rf_model = assets["rf_model"]
kmeans_model = assets["kmeans_model"]
scaler = assets["scaler"]

train_accuracy = assets["train_accuracy"]
test_accuracy = assets["test_accuracy"]
gap_overfitting = assets["gap_overfitting"]

cm = assets["cm"]

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:

    st.markdown("""
    <div style="text-align:center">

    <div style="
    font-size:2rem;
    font-weight:800;
    color:#52796F;
    ">
    🌿 Ruang Tengah
    </div>

    <div style="
    color:#6B7A70;
    margin-bottom:30px;
    ">
    Tempat rehat sejenak buatmu.
    </div>

    </div>
    """, unsafe_allow_html=True)

    menu = option_menu(
        menu_title=None,

        options=[
            "Beranda",
            "Cek Tingkat Stres",
            "Cek Persona Kamu",
            "Dapur Model"
        ],

        icons=[
            "house-door",
            "cloud-haze2",
            "person-badge",
            "cpu"
        ],

        default_index=0,

        styles={

            "container": {
                "padding": "0",
                "background-color": "#DFE7E5"
            },

            "icon": {
                "color": "#52796F",
                "font-size": "18px"
            },

            "nav-link": {
                "font-size": "15px",
                "padding": "12px",
                "border-radius": "10px",
                "color": "#2F3E46",
                "font-weight": "600",
                "--hover-color": "#CFE0D8"
            },

            "nav-link-selected": {
                "background-color": "#84A98C",
                "color": "white"
            }

        }
    )
    

# ==================================================
# BERANDA
# ==================================================
if menu == "Beranda":

    st.markdown("""
    <div class="hero">

    <h1>🌿 Selamat Datang di Ruang Tengah</h1>

    <p>
    Kehidupan sekolah/kampus nggak selalu mudah.
    Kadang tugas menumpuk, jam tidur berantakan,
    dan tekanan ujian datang bersamaan.

    Ruang Tengah hadir untuk membantumu memahami
    pola hidupmu melalui Machine Learning.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================
    # STATISTIK
    # ======================
    st.subheader("📊 Sekilas Tentang Sistem")

    total_data = len(
        pd.read_csv(
            "student-lifestyle-and-stress-dataset.csv"
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Data Pelajar",
            f"{total_data:,}"
        )

    with col2:
        st.metric(
            "Variabel yang Dianalisis",
            "5"
        )

    with col3:
        st.metric(
            "Akurasi Model",
            f"{test_accuracy * 100:.1f}%"
        )

    st.write("")

    # ======================
    # FITUR
    # ======================
    st.subheader("🚀 Apa yang Bisa Kamu Lakukan?")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="custom-card">

        <h3>☁️ Cek Tingkat Stres</h3>

        Prediksi tingkat kerentanan stres
        berdasarkan pola tidur, belajar,
        penggunaan media sosial,
        kehadiran, dan tekanan ujian.

        <br><br>

        <b>Model:</b> Random Forest

        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="custom-card">

        <h3>👥 Cek Persona Pelajar</h3>

        Temukan tipe pelajar yang paling
        mirip dengan kebiasaanmu sehari-hari.

        <br><br>

        <b>Model:</b> K-Means Clustering

        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="custom-card">

        <h3>⚙️ Dapur Model</h3>

        Lihat performa model Machine Learning,
        termasuk validasi terhadap
        kemungkinan overfitting.

        <br><br>

        <b>Evaluasi:</b> Accuracy & Gap Analysis

        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # ======================
    # EDUKASI
    # ======================
    st.subheader("🧠 Kenapa Ini Penting?")

    left, right = st.columns([2, 1])

    with left:

        st.info("""
        **Kesehatan mental pelajar sering kali dipengaruhi oleh:**

        • Kurangnya waktu tidur

        • Tekanan akademik yang tinggi

        • Kebiasaan penggunaan media sosial

        • Ketidakseimbangan antara belajar dan istirahat

        Dengan memahami pola tersebut,
        kita bisa lebih sadar terhadap kondisi diri sendiri.
        """)

    with right:

        st.markdown(f"""
        <div class="custom-card">

        <h3>✨ Fakta Menarik</h3>

        Dataset yang digunakan terdiri dari
        <b>{total_data:,}</b> data pelajar.

        Sistem ini menganalisis kebiasaan
        menggunakan pendekatan Machine Learning.

        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.success(
        "👈 Pilih menu di sebelah kiri untuk mulai mengeksplorasi dirimu."
    )

# ==================================================
# CEK TINGKAT STRES
# ==================================================
elif menu == "Cek Tingkat Stres":

    st.markdown("""
    <div class="hero">

    <h1>☁️ Cek Tingkat Stres</h1>

    <p>
    Isi kebiasaanmu beberapa waktu terakhir.
    Sistem akan memperkirakan tingkat kerentanan
    stres berdasarkan pola hidupmu.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("📝 Isi Kondisimu Saat Ini")

    col1, col2 = st.columns(2)

    with col1:

        sleep = st.slider(
            "🛌 Durasi tidur malam (Jam)",
            2.0,
            10.0,
            6.5
        )

        study = st.slider(
            "📚 Waktu belajar di luar kelas (Jam)",
            0.0,
            15.0,
            4.0
        )

        social = st.slider(
            "📱 Waktu media sosial / game (Jam)",
            0.0,
            10.0,
            3.5
        )

    with col2:

        attendance = st.slider(
            "🏫 Kehadiran sekolah/kampus (%)",
            0.0,
            100.0,
            80.0
        )

        pressure = st.slider(
            "📝 Tekanan tugas / ujian",
            1.0,
            10.0,
            5.0
        )

    st.write("")

    if st.button(
        "🔍 Analisis Kondisiku",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [[
                sleep,
                study,
                social,
                attendance,
                pressure
            ]],

            columns=[
                "Sleep_Hours",
                "Study_Hours",
                "Social_Media_Hours",
                "Attendance",
                "Exam_Pressure"
            ]
        )

        prediction = rf_model.predict(
            input_data
        )[0]

        probability = rf_model.predict_proba(
            input_data
        )[0][1]

        st.divider()

        st.subheader("📊 Hasil Analisis")

        m1, m2, m3 = st.columns(3)

        with m1:

            if prediction == 1:

                st.metric(
                    "Status",
                    "RAWAN STRES"
                )

            else:

                st.metric(
                    "Status",
                    "SEIMBANG"
                )

        with m2:

            st.metric(
                "Probabilitas",
                f"{probability * 100:.1f}%"
            )

        with m3:

            if probability < 0.30:

                level = "Rendah"

            elif probability < 0.70:

                level = "Sedang"

            else:

                level = "Tinggi"

            st.metric(
                "Kategori Risiko",
                level
            )

        st.write("")

        st.markdown(
            "### 📈 Tingkat Kerentanan"
        )

        st.progress(float(probability))

        st.write("")

        # ==========================
        # INTERPRETASI
        # ==========================
        st.markdown(
            "### 💡 Interpretasi Sistem"
        )

        if prediction == 1:

            st.warning(f"""
Kemungkinan stres berada pada kisaran **{probability * 100:.1f}%**.

Sistem mendeteksi bahwa pola hidupmu saat ini
cukup berisiko memicu stres atau burnout.

Cobalah untuk:

• Menjaga jam tidur lebih konsisten

• Memberi waktu istirahat di sela aktivitas

• Mengurangi tekanan yang bisa ditunda

• Mencari dukungan dari teman atau keluarga
            """)

        else:

            st.success(f"""
Kemungkinan stres berada pada kisaran **{probability * 100:.1f}%**.

Secara umum, pola hidupmu masih tergolong
cukup seimbang.

Tetap pertahankan:

• Pola tidur yang cukup

• Ritme belajar yang sehat

• Waktu istirahat yang memadai

• Aktivitas sosial yang tidak berlebihan
            """)

        # ==========================
        # INSIGHT PERSONAL
        # ==========================
        # ==========================
        # INSIGHT PERSONAL BERBASIS AI
        # ==========================
        st.markdown(
            "### 🤖 Rekomendasi Berbasis AI (Simulasi Model)"
        )
        
        # Kita melakukan simulasi jika pengguna mengubah gaya hidupnya
        # Baseline probability adalah 'probability' saat ini (probabilitas stres)
        insights = []
        
        # Simulasi 1: Jika tidur ditingkatkan menjadi minimal 7 jam
        if sleep < 7:
            sim_data = input_data.copy()
            sim_data['Sleep_Hours'] = 7.0
            sim_prob = rf_model.predict_proba(sim_data)[0][1]
            if probability - sim_prob >= 0.02: # Jika turun lebih dari 2%
                insights.append(f"🛌 **Tidur**: Jika kamu menambah jam tidur menjadi 7 jam, model memprediksi kerentanan stresmu akan turun **{ (probability - sim_prob)*100:.1f}%**.")
                
        # Simulasi 2: Jika tekanan akademik (ujian) diturunkan ke level menengah (5)
        if pressure > 5:
            sim_data = input_data.copy()
            sim_data['Exam_Pressure'] = 5.0
            sim_prob = rf_model.predict_proba(sim_data)[0][1]
            if probability - sim_prob >= 0.02:
                insights.append(f"📝 **Tekanan**: Jika kamu bisa mengelola tekanan tugas/ujian ke tingkat wajar, kemungkinan stresmu bisa ditekan turun **{ (probability - sim_prob)*100:.1f}%**.")
                
        # Simulasi 3: Jika waktu medsos dikurangi menjadi 2 jam
        if social > 3:
            sim_data = input_data.copy()
            sim_data['Social_Media_Hours'] = 2.0
            sim_prob = rf_model.predict_proba(sim_data)[0][1]
            if probability - sim_prob >= 0.02:
                insights.append(f"📱 **Media Sosial**: Mengurangi screen time harianmu ke 2 jam diprediksi akan menurunkan risiko stres sebesar **{ (probability - sim_prob)*100:.1f}%**.")
                
        # Simulasi 4: Jika kehadiran kelas ditingkatkan ke 90%
        if attendance < 80:
            sim_data = input_data.copy()
            sim_data['Attendance'] = 90.0
            sim_prob = rf_model.predict_proba(sim_data)[0][1]
            if probability - sim_prob >= 0.02:
                insights.append(f"🏫 **Kehadiran**: Meningkatkan kehadiran kelas di atas 90% dapat mengurangi kerentanan stresmu hingga **{ (probability - sim_prob)*100:.1f}%** (terhindar dari cemas tertinggal materi).")
                
        # Simulasi 5: Jika waktu belajar disesuaikan ke porsi ideal (sekitar 4 jam)
        if study > 8 or study < 2:
            sim_data = input_data.copy()
            sim_data['Study_Hours'] = 4.0
            sim_prob = rf_model.predict_proba(sim_data)[0][1]
            if probability - sim_prob >= 0.02:
                insights.append(f"📚 **Belajar**: Menyesuaikan ritme belajar mandiri ke porsi yang ideal (sekitar 4 jam/hari) diprediksi dapat menurunkan kemungkinan stresmu sebesar **{ (probability - sim_prob)*100:.1f}%**.")
                
        # Tampilkan insights
        if len(insights) > 0:
            st.info("Berdasarkan eksperimen simulasi pada pola gaya hidupmu, model AI menyarankan:")
            for item in insights:
                st.write(f"- {item}")
        else:
            st.success("✨ Saat ini, model tidak menemukan kebiasaan ekstrem yang berdampak besar jika diubah. Pola hidupmu sudah optimal menurut mesin!")

# ==================================================
# CEK PERSONA MAHASISWA
# ==================================================
elif menu == "Cek Persona Kamu":

    st.markdown("""
    <div class="hero">

    <h1>👥 Temukan Persona Mahasiswamu</h1>

    <p>
    Setiap pelajar punya pola hidup yang berbeda.
    Jawab beberapa pertanyaan sederhana di bawah ini
    untuk mengetahui tipe pelajar yang paling mirip denganmu.
    </p>

    </div>
    """, unsafe_allow_html=True)
    st.subheader("📝 Kuis Singkat")

    # ============================================
    # Pertanyaan 1
    # ============================================
    with st.container(border=True):

        st.markdown("""
        <div class="question-title">
        🌙 1. Kalau malam, jam tidurmu biasanya gimana?
        </div>
        """, unsafe_allow_html=True)

        q1 = st.radio(
            "",
            [
                "Sering begadang, tidur di bawah 5 jam",
                "Normal aja sih, sekitar 5–7 jam",
                "Cukup banget, di atas 7 jam"
            ],
            key="q1",
            label_visibility="collapsed"
        )

    sleep_map = {
        "Sering begadang, tidur di bawah 5 jam": 4.0,
        "Normal aja sih, sekitar 5–7 jam": 6.0,
        "Cukup banget, di atas 7 jam": 8.5
    }


    # ============================================
    # Pertanyaan 2
    # ============================================
    with st.container(border=True):

        st.markdown("""
        <div class="question-title">
        📚 2. Kalau di luar jam sekolah/kampus, gaya belajar mandiri kamu gimana?
        </div>
        """, unsafe_allow_html=True)

        q2 = st.radio(
            "",
            [
                "Jarang nyentuh materi (< 2 jam)",
                "Secukupnya aja, sekitar 3–5 jam",
                "Ambis! Di atas 5 jam sehari"
            ],
            key="q2",
            label_visibility="collapsed"
        )

    study_map = {
        "Jarang nyentuh materi (< 2 jam)": 1.5,
        "Secukupnya aja, sekitar 3–5 jam": 4.0,
        "Ambis! Di atas 5 jam sehari": 7.0
    }


    # ============================================
    # Pertanyaan 3
    # ============================================
    with st.container(border=True):

        st.markdown("""
        <div class="question-title">
        📱 3. Screen time buat sosmed/game harianmu?
        </div>
        """, unsafe_allow_html=True)

        q3 = st.radio(
            "",
            [
                "Jarang nyentuh HP (di bawah 2 jam)",
                "Buat hiburan wajar (3–5 jam)",
                "Lumayan candu nih (di atas 5 jam)"
            ],
            key="q3",
            label_visibility="collapsed"
        )

    soc_map = {
        "Jarang nyentuh HP (di bawah 2 jam)": 1.5,
        "Buat hiburan wajar (3–5 jam)": 4.0,
        "Lumayan candu nih (di atas 5 jam)": 7.0
    }


    # ============================================
    # Pertanyaan 4
    # ============================================
    with st.container(border=True):

        st.markdown("""
        <div class="question-title">
        🏫 4. Kehadiran kuliahmu seperti apa?
        </div>
        """, unsafe_allow_html=True)

        q4 = st.radio(
            "",
            [
                "Agak rawan, sering bolos (<70%)",
                "Lumayan aman (70–85%)",
                "Rajin banget (>85%)"
            ],
            key="q4",
            label_visibility="collapsed"
        )

    att_map = {
        "Agak rawan, sering bolos (<70%)": 60.0,
        "Lumayan aman (70–85%)": 75.0,
        "Rajin banget (>85%)": 95.0
    }


    # ============================================
    # Pertanyaan 5
    # ============================================
    with st.container(border=True):

        st.markdown("""
        <div class="question-title">
        📝 5. Saat musim tugas atau ujian, apa yang kamu rasakan?
        </div>
        """, unsafe_allow_html=True)

        q5 = st.radio(
            "",
            [
                "Santai aja, masih bisa di-handle",
                "Lumayan capek dan kepikiran",
                "Berat banget sampai pusing"
            ],
            key="q5",
            label_visibility="collapsed"
        )

    press_map = {
        "Santai aja, masih bisa di-handle": 3.0,
        "Lumayan capek dan kepikiran": 6.0,
        "Berat banget sampai pusing": 9.0
    }

    st.write("")

    if st.button(
        "✨ Temukan Personaku",
        type="primary"
    ):

            input_data_persona = pd.DataFrame(
                [[
                    sleep_map[q1],
                    study_map[q2],
                    soc_map[q3],
                    att_map[q4],
                    press_map[q5]
                ]],

                columns=[
                    "Sleep_Hours",
                    "Study_Hours",
                    "Social_Media_Hours",
                    "Attendance",
                    "Exam_Pressure"
                ]
            )

            input_scaled = scaler.transform(
                input_data_persona
            )

            cluster = kmeans_model.predict(
                input_scaled
            )[0]

            st.divider()

            st.subheader("🌟 Hasil Persona")

            # ======================================
            # PERSONA 1
            # ======================================
            if cluster == 0:

                st.info("""
    ### 😌 Tipe Chill & Santai

    Kamu termasuk pelajar yang cukup menikmati hidup.
    Waktu untuk diri sendiri dan hiburan cukup banyak.

    #### Kelebihan:
    ✅ Mudah beradaptasi

    ✅ Tidak terlalu mudah stres

    ✅ Punya keseimbangan sosial yang baik

    #### Yang perlu diperhatikan:
    ⚠️ Jangan terlalu nyaman sampai melupakan tanggung jawab akademik.

    ⚠️ Tetap pertahankan disiplin belajar.
                """)

                st.progress(0.35)

                st.caption(
                    "Tingkat tekanan akademik cenderung rendah."
                )

            # ======================================
            # PERSONA 2
            # ======================================
            elif cluster == 1:

                st.warning("""
    ### 🔥 Tipe Ambis Under Pressure

    Kamu punya dedikasi tinggi terhadap akademik.
    Targetmu jelas dan kamu berusaha keras mencapainya.

    #### Kelebihan:
    ✅ Sangat disiplin

    ✅ Bertanggung jawab

    ✅ Fokus terhadap tujuan

    #### Yang perlu diperhatikan:
    ⚠️ Rentan mengalami burnout.

    ⚠️ Jangan lupa memberi ruang untuk istirahat.
                """)

                st.progress(0.80)

                st.caption(
                    "Tingkat tekanan akademik cenderung tinggi."
                )

            # ======================================
            # PERSONA 3
            # ======================================
            else:

                st.success("""
    ### 🌿 Tipe Well-Balanced

    Kamu berhasil menjaga keseimbangan antara
    belajar, istirahat, dan kehidupan sosial.

    #### Kelebihan:
    ✅ Manajemen waktu baik

    ✅ Pola hidup relatif sehat

    ✅ Lebih stabil menghadapi tekanan

    #### Yang perlu diperhatikan:
    ⚠️ Pertahankan konsistensi kebiasaan baikmu.
                """)

                st.progress(0.55)

                st.caption(
                    "Keseimbangan hidup berada pada tingkat yang baik."
                )

            st.write("")

            st.subheader("💡 Saran untukmu")

            if cluster == 0:

                st.write(
                    "• Buat jadwal belajar yang lebih konsisten."
                )

                st.write(
                    "• Tetapkan target akademik sederhana setiap minggu."
                )

            elif cluster == 1:

                st.write(
                    "• Jadwalkan waktu istirahat secara rutin."
                )

                st.write(
                    "• Jangan ragu meminta bantuan ketika kewalahan."
                )

            else:

                st.write(
                    "• Pertahankan pola hidup yang sudah baik."
                )

                st.write(
                    "• Tetap evaluasi kondisi diri secara berkala."
                )

# ==================================================
# DAPUR MODEL AI
# ==================================================
elif menu == "Dapur Model":

    st.markdown("""
    <div class="hero">

    <h1>⚙️ Dapur Model AI</h1>

    <p>
    Halaman ini menunjukkan performa model Machine Learning
    yang digunakan pada sistem Ruang Tengah.
    Tujuannya untuk memastikan model bekerja dengan baik
    dan tidak mengalami overfitting.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Ringkasan Performa Model")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Akurasi Data Latih",
            f"{train_accuracy:.3f}"
        )

    with col2:

        st.metric(
            "Akurasi Data Uji",
            f"{test_accuracy:.3f}"
        )

    with col3:

        st.metric(
            "Gap Overfitting",
            f"{gap_overfitting:.3f}"
        )

    st.write("")

    st.subheader("🔍 Evaluasi Stabilitas Model")

    if gap_overfitting >= 0.05:

        st.error(f"""
### 🚨 Indikasi Overfitting

Model memiliki selisih akurasi sebesar **{gap_overfitting:.3f}**.

Hal ini menunjukkan model terlalu menghafal data latih
sehingga performanya berpotensi menurun
saat digunakan pada data baru.

#### Dampak:
• Generalisasi model kurang baik

• Prediksi dapat menjadi kurang stabil

• Model perlu dilakukan tuning ulang
        """)

        stability_score = max(
            0,
            1 - (gap_overfitting * 5)
        )

    else:

        st.success(f"""
### ✅ Model Stabil

Model memiliki selisih akurasi sebesar **{gap_overfitting:.3f}**.

Perbedaan akurasi yang kecil menunjukkan bahwa
model mampu mempelajari pola secara sehat
tanpa sekadar menghafal data latih.

#### Keunggulan:
• Generalisasi lebih baik

• Prediksi lebih konsisten

• Siap digunakan pada data baru
        """)

        stability_score = 1 - gap_overfitting

    st.write("")

    st.subheader("🌿 Tingkat Kesehatan Model")

    st.progress(
        float(stability_score)
    )

    st.caption(
        f"Skor stabilitas model: {stability_score * 100:.1f}%"
    )

    st.write("")

    # =========================================
    # CONFUSION MATRIX SUMMARY
    # =========================================
    st.subheader("🧩 Confusion Matrix Summary")

    safe_correct = cm[0][0]
    stress_correct = cm[1][1]

    total_correct = (
        safe_correct +
        stress_correct
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(f"""
        <div class="custom-card">

        <h3>✅ Prediksi Aman</h3>

        Sistem berhasil mengenali
        <h2>{safe_correct}</h2>

        pelajar yang memang
        berada pada kondisi aman.

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="custom-card">

        <h3>⚠️ Prediksi Stres</h3>

        Sistem berhasil mengenali
        <h2>{stress_correct}</h2>

        pelajar yang memang
        berada pada kondisi stres.

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("📚 Teknologi yang Digunakan")

    tech1, tech2 = st.columns(2)

    with tech1:

        st.markdown("""
        <div class="custom-card">

        <h3>🌲 Random Forest</h3>

        Digunakan untuk memprediksi
        tingkat stres pelajar.

        <br>

        Random Forest bekerja dengan
        menggabungkan banyak pohon keputusan
        untuk menghasilkan prediksi
        yang lebih stabil.

        </div>
        """, unsafe_allow_html=True)

    with tech2:

        st.markdown("""
        <div class="custom-card">

        <h3>🎯 K-Means Clustering</h3>

        Digunakan untuk menemukan
        kelompok persona pelajar.

        <br>

        Algoritma ini mengelompokkan
        pelajar berdasarkan
        kemiripan pola hidup mereka.

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("🏁 Kesimpulan")

    if gap_overfitting < 0.05:

        st.success("""
Model Machine Learning yang digunakan
memiliki performa yang baik dan stabil.

Berdasarkan evaluasi yang dilakukan,
model mampu memberikan prediksi yang
cukup andal untuk membantu pelajar
memahami kondisi diri mereka.
        """)

    else:

        st.warning("""
Model masih menunjukkan gejala overfitting.

Perlu dilakukan optimasi lebih lanjut,
seperti tuning hyperparameter
atau pengembangan dataset,
agar performa model menjadi lebih baik.
        """)