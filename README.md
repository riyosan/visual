# 🗺️ Visualisasi Anomali Absensi

Sistem deteksi anomali lokasi absensi pegawai menggunakan **DBSCAN**, **ST-DBSCAN**, dan **Haversine Distance**, divisualisasikan dengan **Streamlit + Folium + Plotly**.

---

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi | Alasan |
|---|---|---|
| **Preprocessing** | Python (Google Colab) | Mudah dijalankan, gratis GPU/RAM |
| **Visualisasi Web** | Streamlit | Native Python, mudah deploy |
| **Peta Interaktif** | Folium + streamlit-folium | Peta OpenStreetMap, marker, heatmap, cluster |
| **Peta 3D** | PyDeck | Visualisasi 3D berbasis deck.gl |
| **Chart** | Plotly | Chart interaktif (pie, bar, scatter, histogram) |
| **Clustering** | scikit-learn DBSCAN | Spatial + ST-DBSCAN |
| **Jarak** | Haversine | Jarak akurat di permukaan bumi |

---

## 📁 Struktur File

```
├── preprocessing_absensi.ipynb   # Notebook Google Colab (preprocessing)
├── app.py                        # Streamlit app (visualisasi)
├── requirements.txt              # Dependencies Python
├── README.md                     # Dokumentasi ini
└── dataset_absensi_final2.xlsx   # Dataset asli (input)
```

---

## 🚀 Cara Penggunaan

### Step 1: Preprocessing di Google Colab

1. Upload `preprocessing_absensi.ipynb` ke Google Colab
2. Upload `dataset_absensi_final2.xlsx` ke Colab (atau mount Google Drive)
3. Sesuaikan `FILE_PATH` di cell "Load Dataset"
4. Jalankan semua cell secara berurutan
5. Download hasil `absensi_processed.csv`

### Step 2: Jalankan Streamlit App

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan app
streamlit run app.py
```

### Step 3: Upload Data ke App

- Letakkan `absensi_processed.csv` di folder yang sama dengan `app.py`, **ATAU**
- Upload melalui sidebar kiri di aplikasi

---

## 🔄 Pipeline Preprocessing

```
Raw Data (Excel)
    ↓
[STEP 1] Time Transform
    → timestamp_num, jam, menit, jam_desimal, weekday
    ↓
[STEP 2] Coordinate Transform
    → lat_rad, long_rad (untuk DBSCAN haversine)
    → validasi koordinat Indonesia
    ↓
[STEP 3] DBSCAN Spatial (per SKPD)
    → cluster_masuk, cluster_pulang
    → is_noise_masuk, is_noise_pulang
    → cluster_size_masuk, cluster_size_pulang
    ↓
[STEP 4] Estimasi Centroid Kantor
    → office_lat, office_long (per SKPD)
    → Ambil cluster terbesar → hitung centroid
    ↓
[STEP 5] Haversine Distance
    → dist_km (jarak absensi ke kantor)
    → outside_300m, very_far, extreme_far
    ↓
[STEP 6] Validation Logic
    → no_note, far_no_note, far_with_note
    → near_but_status0, far_but_status1
    ↓
[STEP 7] ST-DBSCAN (Spatio-Temporal)
    → st_cluster_masuk, st_cluster_pulang
    → is_st_noise_masuk, is_st_noise_pulang
    ↓
[STEP 8] Anomaly Scoring
    → anomaly_score (0-N)
    → risk_level (LOW / MED / HIGH)
    ↓
Output: absensi_processed.csv
```

---

## 📊 Anomaly Scoring

| Sinyal | Bobot | Keterangan |
|---|---|---|
| `extreme_far` (>50km) | +3 | Indikasi fake GPS kuat |
| `very_far` (>5km) | +2 | Sangat jauh dari kantor |
| `far_no_note` | +2 | Jauh + tidak ada alasan |
| `outside_300m` | +1 | Di luar radius 300m |
| `is_noise_masuk` | +1 | Tidak masuk cluster manapun |
| `is_st_noise_masuk` | +1 | Tidak konsisten waktu+lokasi |
| `far_but_status1` | +1 | Mismatch sistem lama |
| `far_with_note` | -1 | Ada alasan (tugas luar) |

### Risk Level

| Score | Risk Level |
|---|---|
| 0 – 1 | 🟢 LOW |
| 2 – 3 | 🟡 MED |
| ≥ 4 | 🔴 HIGH |

---

## 🗺️ Fitur Visualisasi

### Tab Overview
- Metrics: total absensi, karyawan, distribusi risk
- Pie chart distribusi risk level
- Bar chart risk per SKPD
- Histogram anomaly score

### Tab Peta
- **Marker per Risk**: titik berwarna merah/oranye/hijau
- **Cluster Marker**: pengelompokan otomatis
- **Heatmap**: kepadatan anomali
- **Peta 3D (PyDeck)**: visualisasi 3D interaktif
- Marker kantor SKPD + lingkaran radius 300m
- Popup detail per titik absensi

### Tab Temporal
- Distribusi absensi per jam
- Distribusi per hari dalam minggu
- Trend anomali harian
- Scatter: jam vs jarak ke kantor

### Tab Jarak
- Histogram distribusi jarak
- Box plot per SKPD
- Tabel kasus extreme (>5km)

### Tab Karyawan
- Top 10 karyawan HIGH risk
- Scatter: total absensi vs anomaly score
- Tabel karyawan berisiko tinggi

### Tab Cluster
- Statistik noise DBSCAN masuk/pulang
- Distribusi ukuran cluster
- Analisis ST-DBSCAN noise

### Tab Data
- Tabel lengkap dengan filter
- Search karyawan ID
- Download CSV

---

## ⚙️ Konfigurasi Parameter

Di notebook `preprocessing_absensi.ipynb`, Anda bisa menyesuaikan:

```python
# DBSCAN Spatial
EPS_KM      = 0.1   # Radius cluster (km) - default 100m
MIN_SAMPLES = 3     # Minimum titik per cluster

# ST-DBSCAN
ST_EPS_KM    = 0.1  # Radius spasial (km)
ST_EPS_HOURS = 1.0  # Radius temporal (jam)
ST_MIN_SAMP  = 3    # Minimum titik
```

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
folium>=0.15.0
streamlit-folium>=0.18.0
plotly>=5.18.0
pydeck>=0.8.0
haversine>=2.8.0
openpyxl>=3.1.0
```
