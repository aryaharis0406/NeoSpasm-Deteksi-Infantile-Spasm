# NeoSpasm — App Skrining Spasme Infantil (multi-halaman)

**ALAT SKRINING / TRIASE — BUKAN DIAGNOSIS.** Fusi 3-arah: RGB R(2+1)D (crop bayi) + PoseTCN + VideoMAE (bobot 0,1 / 0,3 / 0,6).

## Struktur
```
Beranda.py                 <- halaman utama (judul + penjelasan)
pages/1_🔍_Prediksi.py     <- halaman input video + hasil
neospasm_core.py           <- logika model & tema (dipakai bersama)
.streamlit/config.toml     <- tema warna
models/                    <- 3 file model (.pt)
```

## 1. Siapkan model
Salin dari Google Drive `SEC 2026/Modelling/runs/` ke folder `models/`:
```
models/model_exp_dense.pt        <- runs/exp_dense_.../
models/model_pose_dense.pt       <- runs/pose_dense_.../
models/model_videomae_exp1.pt    <- runs/videomae_exp1_.../
```

## 2. Install & jalankan
```bash
pip install -r requirements.txt
streamlit run Beranda.py
```
Menu halaman (Beranda / Prediksi) muncul otomatis di sidebar kiri.
Pertama jalan akan mengunduh VideoMAE (~350MB) & YOLO otomatis.

## Catatan
- Sensitivitas tinggi (ambang default 0,45) demi keamanan skrining.
- Performa jujur (validasi lintas-subjek): video-AUC ~0,84. Bukan perangkat medis.
