"""Halaman Prediksi — NeoSpasm"""
import os, sys, tempfile
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neospasm_core as core

st.set_page_config(page_title="NeoSpasm — Prediksi", page_icon="🔍", layout="centered")
core.inject_css()

st.markdown("""
<div class="hero" style="padding:26px 30px;">
  <div class="badge">🔍 HALAMAN PREDIKSI</div>
  <h1 style="font-size:2rem;">Analisis Video</h1>
  <p>Unggah video bayi, lalu jalankan analisis untuk melihat tingkat kekhawatiran.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer" style="margin:10px 0;">
  <b>⚠️ Bukan diagnosis.</b> Alat skrining. Apa pun hasilnya, bila ragu segera periksakan bayi ke dokter.
</div>
""", unsafe_allow_html=True)

# ---- sidebar setelan ----
with st.sidebar:
    st.markdown("### ⚙️ Setelan")
    thr = st.slider("Ambang perhatian", 0.20, 0.80, 0.45, 0.05,
                    help="Lebih rendah = lebih sensitif (lebih mudah memberi peringatan).")
    st.caption("Default 0,45 — disetel sensitivitas tinggi demi keamanan skrining.")

missing = core.models_missing()
if missing:
    st.warning("Berkas model belum tersedia di folder `models/`:\n\n"
               + "\n".join(f"- `{m}`" for m in missing)
               + "\n\nSalin dari Google Drive (folder run) lalu jalankan ulang.")

# ---- uploader ----
st.markdown('<div class="card">', unsafe_allow_html=True)
up = st.file_uploader("📤 Pilih video bayi", type=["mp4","mov","avi","mkv"],
                      label_visibility="visible")
st.markdown('</div>', unsafe_allow_html=True)

if up is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tf:
        tf.write(up.read()); vpath = tf.name
    col_v, col_b = st.columns([3, 2])
    with col_v:
        st.video(vpath)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("🔍 Jalankan Prediksi", use_container_width=True, disabled=bool(missing))
        st.caption("Analisis butuh beberapa saat (3 model berjalan).")

    if not missing and go:
        M = core.load_all()
        prog = st.progress(0.0, text="Mulai...")
        wins, dur = core.analyze_video(M, vpath, prog)
        prog.empty()

        fused = [w["fused"] for w in wins]
        score = max(fused); concern = round(score*100)
        positif = score >= thr
        color = "#ef4444" if positif else "#22c55e"
        label = "IS terdeteksi (tanda menyerupai)" if positif else "IS tidak terdeteksi"

        # ----- METER hasil -----
        st.markdown(f"""
        <div class="card">
          <div class="meter">
            <div class="num" style="color:{color};">{concern}%</div>
            <div class="lab">Tingkat kekhawatiran spasme infantil</div>
            <div class="bar"><div class="fill" style="width:{concern}%;background:{color};"></div></div>
            <div style="margin-top:10px;">
              <span class="pill" style="background:{color}22;color:{color};">{label}</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ----- VERDICT -----
        if positif:
            st.markdown("""
            <div class="verdict v-alert">
              <b>⚠️ Ditemukan momen yang menyerupai pola spasme infantil.</b><br>
              Mohon <b>segera periksakan bayi ke dokter anak / neurolog anak</b>. Bawa video ini
              sebagai bahan diskusi. Ingat: <b>ini bukan diagnosis</b>, hanya skrining awal.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verdict v-ok">
              <b>Tidak ditemukan pola spasme yang jelas.</b><br>
              <b>Ini BUKAN jaminan bayi sehat.</b> Jika Anda tetap melihat gerakan mencurigakan
              (kepala mengangguk / tubuh menekuk berulang dalam kelompok), <b>tetap periksakan ke dokter.</b>
            </div>""", unsafe_allow_html=True)

        # ----- TIMELINE -----
        st.markdown("#### 📈 Lini masa per jendela (5 detik)")
        try:
            import pandas as pd
            df = pd.DataFrame({"detik": [round(w["t0"],1) for w in wins],
                               "skor fusi": [round(w["fused"],3) for w in wins]}).set_index("detik")
            st.bar_chart(df, color=color)
        except Exception:
            pass

        # ----- RINCIAN MODEL -----
        wmax = max(wins, key=lambda w: w["fused"])
        with st.expander("🔬 Rincian per model (jendela paling mengkhawatirkan)"):
            a, b, c = st.columns(3)
            a.metric("RGB R(2+1)D", f"{wmax['rgb']*100:.0f}%")
            b.metric("Pose", f"{wmax['pose']*100:.0f}%")
            c.metric("VideoMAE", f"{wmax['vmae']*100:.0f}%")
            st.caption(f"Jendela detik {wmax['t0']:.1f}–{wmax['t1']:.1f} → fusi {wmax['fused']*100:.0f}% "
                       f"(bobot 0,1 / 0,3 / 0,6).")

st.markdown("<p style='text-align:center;color:#94a3b8;font-size:.82rem;margin-top:24px;'>"
            "NeoSpasm — alat skrining riset, bukan perangkat medis.</p>", unsafe_allow_html=True)
