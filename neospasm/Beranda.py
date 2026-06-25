"""Beranda — NeoSpasm"""
import streamlit as st
import neospasm_core as core

st.set_page_config(page_title="NeoSpasm — Beranda", page_icon="🩺", layout="centered")
core.inject_css()

# ---------- HERO ----------
st.markdown("""
<div class="hero">
  <div class="badge">🩺 SKRINING · BUKAN DIAGNOSIS</div>
  <h1>NeoSpasm</h1>
  <p>Deteksi dini <b>Spasme Infantil</b> pada bayi berbasis analisis video —
  membantu orang tua & tenaga kesehatan mengenali tanda lebih cepat.</p>
</div>
""", unsafe_allow_html=True)

# ---------- TENTANG APLIKASI ----------
st.markdown("""
<div class="card">
  <h3>Tentang aplikasi ini</h3>
  <p style="color:#475569; line-height:1.6;">
  NeoSpasm adalah aplikasi <b>skrining berbasis kecerdasan buatan</b> yang menganalisis
  video gerakan bayi untuk mengenali pola yang menyerupai <b>Spasme Infantil (Infantile Spasms / IS)</b>—
  sejenis epilepsi langka pada bayi yang penanganannya harus cepat. Anda cukup mengunggah
  video, dan aplikasi memberi <b>perkiraan tingkat kekhawatiran</b> beserta saran tindak lanjut.
  </p>
  <p style="color:#475569; line-height:1.6;">
  Aplikasi memadukan <b>tiga model AI</b> sekaligus (analisis penampilan gerak,
  kerangka tubuh/pose, dan model fondasi video) agar penilaian lebih terpercaya.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------- APA ITU IS ----------
st.markdown("""
<div class="card">
  <h3>Apa itu Spasme Infantil?</h3>
  <p style="color:#475569; line-height:1.6;">
  Spasme infantil adalah kejang singkat (1–2 detik) berupa <b>tubuh menekuk, kepala mengangguk,
  atau lengan terangkat</b>, sering muncul <b>berkelompok</b> dan biasanya pada bayi usia 3–12 bulan.
  Karena singkat, gejalanya mudah terlewat atau dikira gerakan biasa. <b>Deteksi & pengobatan
  yang terlambat dapat berdampak pada perkembangan bayi</b>—itulah mengapa pengenalan dini penting.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------- CARA KERJA ----------
st.markdown("<h3 style='margin:18px 0 4px 4px;'>Cara kerja</h3>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="step"><div class="ico">📤</div><b>1. Unggah video</b>'
                '<p>Rekam atau pilih video gerakan bayi, lalu unggah di halaman Prediksi.</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="step"><div class="ico">🧠</div><b>2. Analisis AI</b>'
                '<p>Video dipindai per jendela waktu; tiga model menilai pola gerakan.</p></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="step"><div class="ico">📊</div><b>3. Hasil & saran</b>'
                '<p>Muncul tingkat kekhawatiran (%) dan anjuran untuk periksa ke dokter.</p></div>',
                unsafe_allow_html=True)

# ---------- CTA ----------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="card" style="text-align:center; background:linear-gradient(135deg,#f0f9ff,#ecfeff);">
  <h3 style="margin-bottom:6px;">Siap mencoba?</h3>
  <p style="color:#475569;">Buka halaman <b>🔍 Prediksi</b> di menu samping untuk mengunggah video.</p>
</div>
""", unsafe_allow_html=True)

# ---------- DISCLAIMER ----------
st.markdown("""
<div class="disclaimer">
  <b>⚠️ Penting:</b> NeoSpasm adalah alat <b>bantu skrining riset</b>, <b>bukan alat diagnosis</b>
  dan bukan perangkat medis. Hasil apa pun—termasuk "tidak terdeteksi"—<b>tidak menggantikan
  pemeriksaan dokter</b>. Bila Anda mencurigai bayi mengalami spasme, <b>segera temui dokter anak
  atau neurolog anak</b>, berapa pun hasil aplikasi ini.
</div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;color:#94a3b8;font-size:.82rem;margin-top:20px;'>"
            "NeoSpasm — Proyek SEC Satria Data 2026</p>", unsafe_allow_html=True)
