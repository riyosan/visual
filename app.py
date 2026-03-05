"""
Visualisasi & Deteksi Anomali Absensi — FULL INTEGRATED APP
Fitur baru:
  - Upload data sudah diproses → langsung visualisasi
  - Auto-detect & fix separator koma desimal (locale Indonesia)
  - approver_status: analisis false negative, TOLAK, pending lama
  - Menu 🎯 Hunting terintegrasi penuh
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, AntPath
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import io
import hashlib
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Deteksi Anomali Absensi",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:bold;color:#1f77b4;text-align:center;padding:1rem 0 .2rem}
.sub-header{text-align:center;color:#666;font-size:.95rem;margin-bottom:1.5rem}
.step-box{background:#f8f9fa;border-left:4px solid #1f77b4;padding:.8rem 1rem;border-radius:0 8px 8px 0;margin-bottom:.8rem}
.step-title{font-weight:bold;color:#1f77b4;font-size:1rem}
.step-desc{color:#444;font-size:.88rem;margin-top:.2rem}

/* Hunting */
.hunt-header{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.5rem;border:1px solid #e94560;position:relative;overflow:hidden}
.hunt-header::before{content:'🎯';position:absolute;right:2rem;top:50%;transform:translateY(-50%);font-size:4rem;opacity:.12}
.hunt-title{font-size:1.7rem;font-weight:900;color:#e94560;font-family:'Courier New',monospace;letter-spacing:2px;margin:0}
.hunt-sub{color:#a8b2d8;font-size:.83rem;margin-top:.3rem;font-family:'Courier New',monospace}
.section-header{display:flex;align-items:center;gap:.8rem;background:#f8f9fa;border-left:5px solid #1f77b4;padding:.8rem 1.2rem;border-radius:0 10px 10px 0;margin:1.2rem 0 1rem}
.section-icon{font-size:1.5rem}
.section-title{font-size:1.1rem;font-weight:700;color:#2c3e50;margin:0}
.section-desc{font-size:.78rem;color:#7f8c8d;margin:0}
.metric-grid{display:flex;gap:.8rem;flex-wrap:wrap;margin:.8rem 0}
.metric-card{flex:1;min-width:110px;background:white;border:1px solid #e0e0e0;border-radius:10px;padding:.9rem;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.06)}
.metric-val{font-size:1.45rem;font-weight:800;color:#2c3e50}
.metric-lbl{font-size:.68rem;color:#7f8c8d;margin-top:2px;text-transform:uppercase;letter-spacing:.4px}
.metric-card.danger .metric-val{color:#e74c3c}
.metric-card.warning .metric-val{color:#f39c12}
.metric-card.success .metric-val{color:#27ae60}
.metric-card.info .metric-val{color:#3498db}

/* Approver badges */
.badge-tolak{background:#e74c3c;color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold}
.badge-terima{background:#27ae60;color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold}
.badge-pending{background:#95a5a6;color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold}
.badge-high{background:#e74c3c;color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold}

.alert-box{border-radius:8px;padding:.9rem 1.1rem;margin:.6rem 0;font-size:.88rem}
.alert-red{background:#fff5f5;border:1px solid #fc8181;color:#c53030}
.alert-orange{background:#fffaf0;border:1px solid #f6ad55;color:#c05621}
.alert-gray{background:#f7fafc;border:1px solid #cbd5e0;color:#4a5568}

.watchlist-item{display:flex;align-items:center;gap:.7rem;background:white;border:1px solid #fde8e8;border-left:4px solid #e74c3c;border-radius:7px;padding:.6rem .9rem;margin-bottom:.4rem;font-size:.85rem}
.trend-up{color:#e74c3c;font-weight:bold}
.trend-down{color:#27ae60;font-weight:bold}
.trend-flat{color:#95a5a6}
.timeline-tip{background:#eef2ff;border-left:3px solid #667eea;padding:.45rem .8rem;border-radius:0 6px 6px 0;font-size:.8rem;color:#4a5568;margin-bottom:.8rem}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def haversine_scalar(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def risk_color_folium(r): return {'HIGH':'red','MED':'orange','LOW':'green'}.get(r,'gray')
def risk_color_hex(r):    return {'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'}.get(r,'#95a5a6')

def approver_badge(v):
    if pd.isna(v) or str(v).strip()=='': return "<span class='badge-pending'>PENDING</span>"
    v = str(v).strip().upper()
    if 'TOLAK' in v: return "<span class='badge-tolak'>TOLAK</span>"
    if 'TERIMA' in v: return "<span class='badge-terima'>TERIMA</span>"
    return f"<span class='badge-pending'>{v}</span>"

def _df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# ============================================================
# AUTO-FIX DECIMAL SEPARATOR (koma → titik, locale Indonesia)
# ============================================================
def fix_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deteksi kolom yang seharusnya numerik tapi terbaca string
    karena separator desimal koma (locale Indonesia Excel).
    Contoh: '3,5722868' → 3.5722868
    """
    numeric_target = [
        'lat','long','lat_rad','long_rad',
        'office_lat','office_long','dist_km',
        'jarak','jam_desimal',
        'anomaly_score','cluster_size_masuk','cluster_size_pulang',
        'timestamp_num','jam','menit','weekday',
        'outside_300m','no_note','far_no_note','far_with_note',
        'near_but_status0','very_far','extreme_far',
        'is_noise_masuk','is_noise_pulang',
        'is_st_noise_masuk','is_st_noise_pulang',
        'cluster_masuk','cluster_pulang',
        'st_cluster_masuk','st_cluster_pulang',
        'status_lokasi',
    ]
    fixed = []
    for col in df.columns:
        if col in numeric_target and df[col].dtype == object:
            try:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(',', '.', regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors='coerce')
                fixed.append(col)
            except Exception:
                pass
        elif col not in numeric_target and df[col].dtype == object:
            # Coba cek apakah kolom ini sebenarnya numerik
            sample = df[col].dropna().head(20).astype(str)
            if sample.str.match(r'^-?\d+,\d+$').mean() > 0.7:
                try:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',','.', regex=False),
                        errors='coerce'
                    )
                    fixed.append(col)
                except Exception:
                    pass
    return df, fixed

# ============================================================
# LOAD & VALIDATE (data sudah diproses)
# ============================================================
@st.cache_data(show_spinner=False)
def load_processed_file(file_bytes: bytes, file_name: str):
    buf = io.BytesIO(file_bytes)
    if file_name.endswith('.csv'):
        # Coba berbagai separator
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(buf, sep=sep)
                if len(df.columns) > 3:
                    break
                buf.seek(0)
            except Exception:
                buf.seek(0)
    else:
        df = pd.read_excel(buf)

    # Fix decimal
    df, fixed_cols = fix_decimal_columns(df)

    # Parse datetime
    if 'tanggal_kirim' in df.columns:
        df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'], errors='coerce')

    # Rebuild feature columns jika belum ada
    if 'tanggal_kirim' in df.columns:
        if 'jam' not in df.columns:
            df['jam'] = df['tanggal_kirim'].dt.hour
        if 'menit' not in df.columns:
            df['menit'] = df['tanggal_kirim'].dt.minute
        if 'jam_desimal' not in df.columns:
            df['jam_desimal'] = df['jam'] + df['menit']/60.0
        if 'weekday' not in df.columns:
            df['weekday'] = df['tanggal_kirim'].dt.weekday
        if 'tanggal' not in df.columns:
            df['tanggal'] = df['tanggal_kirim'].dt.date

    # Normalize jenis
    if 'jenis' in df.columns:
        df['jenis'] = df['jenis'].astype(str).str.strip().str.upper()

    # Normalize approver_status
    if 'approver_status' in df.columns:
        df['approver_status'] = df['approver_status'].astype(str).str.strip()
        df['approver_status'] = df['approver_status'].replace({'nan':'', 'None':'', 'NaN':''})

    # Tambah kolom approver flags
    if 'approver_status' in df.columns:
        df['is_tolak']   = df['approver_status'].str.upper().str.contains('TOLAK', na=False).astype(int)
        df['is_terima']  = df['approver_status'].str.upper().str.contains('TERIMA', na=False).astype(int)
        df['is_pending'] = ((df['approver_status'] == '') | df['approver_status'].isna()).astype(int)
        # False negative: TERIMA tapi HIGH risk
        if 'risk_level' in df.columns:
            df['false_negative'] = ((df['is_terima'] == 1) & (df['risk_level'] == 'HIGH')).astype(int)
        else:
            df['false_negative'] = 0

    # Rebuild risk_level jika tidak ada
    if 'risk_level' not in df.columns and 'anomaly_score' in df.columns:
        def rl(s):
            if s >= 70: return 'HIGH'
            elif s >= 30: return 'MED'
            return 'LOW'
        df['risk_level'] = df['anomaly_score'].apply(rl)

    # system_action
    if 'system_action' not in df.columns and 'risk_level' in df.columns:
        def sa(r):
            if r == 'LOW': return 'AUTO APPROVE'
            elif r == 'MED': return 'HOLD (Perlu Review)'
            return 'TEMP REJECT + NOTIF APPROVER'
        df['system_action'] = df['risk_level'].apply(sa)

    return df, fixed_cols

def build_office_centroid(df: pd.DataFrame):
    """Rekonstruksi office_centroid dari kolom office_lat/office_long."""
    if 'office_lat' in df.columns and 'id_skpd' in df.columns:
        oc = (df.groupby('id_skpd')[['office_lat','office_long']]
              .first().reset_index())
        return oc
    return pd.DataFrame(columns=['id_skpd','office_lat','office_long'])

def validate_dataframe(df: pd.DataFrame):
    """Return list of warning strings."""
    warns = []
    required = ['karyawan_id','lat','long','tanggal_kirim','jenis','id_skpd']
    missing = [c for c in required if c not in df.columns]
    if missing:
        warns.append(f"Kolom wajib tidak ditemukan: {missing}")
    if 'lat' in df.columns:
        n_bad = (~df['lat'].between(-90,90)).sum()
        if n_bad > 0:
            warns.append(f"{n_bad} baris koordinat lat di luar range valid")
    if 'risk_level' not in df.columns:
        warns.append("Kolom risk_level tidak ditemukan — akan dihitung ulang dari anomaly_score")
    return warns

# ============================================================
# FILTER HELPER
# ============================================================
@st.cache_data(show_spinner=False)
def apply_filters(df_hash, df, skpd, risk_tuple, jenis_tuple, date_range, dist_range, approver_filter):
    f = df.copy()
    if skpd != 'Semua':
        f = f[f['id_skpd'] == skpd]
    if risk_tuple:
        f = f[f['risk_level'].isin(list(risk_tuple))]
    if jenis_tuple:
        f = f[f['jenis'].isin(list(jenis_tuple))]
    if date_range and len(date_range) == 2 and 'tanggal_kirim' in f.columns:
        f = f[(f['tanggal_kirim'].dt.date >= date_range[0]) &
              (f['tanggal_kirim'].dt.date <= date_range[1])]
    if 'dist_km' in f.columns:
        f = f[(f['dist_km'] >= dist_range[0]) & (f['dist_km'] <= dist_range[1])]
    if approver_filter and 'approver_status' in f.columns:
        if approver_filter == 'TOLAK':
            f = f[f['is_tolak'] == 1]
        elif approver_filter == 'TERIMA':
            f = f[f['is_terima'] == 1]
        elif approver_filter == 'PENDING':
            f = f[f['is_pending'] == 1]
        elif approver_filter == 'False Negative':
            f = f[f['false_negative'] == 1]
    return f

# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    st.sidebar.markdown("## 🗺️ Deteksi Anomali Absensi")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "📌 Navigasi",
        ["🏠 Beranda", "📥 Upload Data", "📊 Visualisasi", "🎯 Hunting", "🔮 Prediksi"],
        index=0
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📂 Upload Data (sudah diproses)")
    uploaded = st.sidebar.file_uploader(
        "Upload CSV / Excel hasil preprocessing",
        type=['csv','xlsx'],
        help="Upload file absensi yang sudah memiliki kolom risk_level, dist_km, dll."
    )

    filters = {}
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        with st.sidebar.expander("🔍 Filter Data", expanded=True):
            skpd_list = ['Semua'] + sorted(df['id_skpd'].unique().tolist())
            filters['skpd']   = st.selectbox("SKPD", skpd_list)
            filters['risk']   = st.multiselect("Risk Level", ['HIGH','MED','LOW'],
                                               default=['HIGH','MED','LOW'])
            filters['jenis']  = st.multiselect("Jenis", ['M','P'], default=['M','P'],
                                               format_func=lambda x: 'Masuk' if x=='M' else 'Pulang')
            if 'tanggal_kirim' in df.columns:
                min_d = df['tanggal_kirim'].min().date()
                max_d = df['tanggal_kirim'].max().date()
                filters['date'] = st.date_input("Rentang Tanggal",
                                                value=(min_d, max_d),
                                                min_value=min_d, max_value=max_d)
            else:
                filters['date'] = None
            max_dist = float(df['dist_km'].max()) if 'dist_km' in df.columns else 100.0
            filters['dist'] = st.slider("Jarak ke Kantor (km)", 0.0,
                                        min(max_dist, 100.0),
                                        (0.0, min(max_dist, 100.0)), 0.1)
            if 'approver_status' in df.columns:
                filters['approver'] = st.selectbox(
                    "Approver Status",
                    ['Semua','TERIMA','TOLAK','PENDING','False Negative'],
                    help="False Negative = TERIMA tapi HIGH risk"
                )
            else:
                filters['approver'] = 'Semua'

        with st.sidebar.expander("🗺️ Pengaturan Peta", expanded=False):
            filters['map_type'] = st.radio(
                "Tipe Peta",
                ['marker','cluster','heatmap'],
                format_func=lambda x: {'marker':'📍 Marker','cluster':'🔵 Cluster','heatmap':'🔥 Heatmap'}[x]
            )

        # Watchlist
        wl = st.session_state.get('watchlist', [])
        if wl:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👁️ Watchlist")
            for emp_id in wl:
                ed = df[df['karyawan_id'] == emp_id]
                nh = (ed['risk_level']=='HIGH').sum() if not ed.empty else 0
                st.sidebar.markdown(f"""
                <div class='watchlist-item'>
                    <span>🔴</span><span><b>ID {emp_id}</b> — {nh} HIGH</span>
                </div>""", unsafe_allow_html=True)
            if st.sidebar.button("🗑️ Clear Watchlist"):
                st.session_state['watchlist'] = []
                st.rerun()

    return page, uploaded, filters

# ============================================================
# PAGE: BERANDA
# ============================================================
def page_beranda():
    st.markdown('<div class="main-header">🗺️ Deteksi Anomali Absensi Pegawai</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload data yang sudah diproses → langsung visualisasi & investigasi</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.info("### 📥 Step 1\n**Upload Data**\nUpload file Excel/CSV hasil preprocessing.")
    with col2: st.success("### 📊 Step 2\n**Visualisasi**\nPeta, temporal, jarak, distribusi risk.")
    with col3: st.warning("### 🎯 Step 3\n**Hunting**\nInvestigasi mendalam per pegawai, SKPD, atau tanggal.")
    with col4: st.error("### 🔮 Step 4\n**Prediksi**\nSkor absensi baru berdasarkan referensi kantor.")

    st.markdown("---")
    st.markdown("### 📋 Kolom yang Dibutuhkan")
    cols_info = pd.DataFrame([
        ['karyawan_id','integer','ID pegawai','Wajib'],
        ['id_skpd','integer','ID kantor/SKPD','Wajib'],
        ['lat / long','float','Koordinat absensi','Wajib'],
        ['tanggal_kirim','datetime','Waktu absensi','Wajib'],
        ['jenis','M / P','Masuk atau Pulang','Wajib'],
        ['risk_level','LOW/MED/HIGH','Output anomali','Wajib'],
        ['anomaly_score','integer','Skor risiko','Wajib'],
        ['dist_km','float','Jarak ke kantor (km)','Wajib'],
        ['approver_status','TERIMA/TOLAK','Keputusan atasan','Opsional'],
        ['catatan','string','Alasan tugas luar','Opsional'],
        ['office_lat/long','float','Koordinat kantor SKPD','Opsional'],
    ], columns=['Kolom','Tipe/Value','Keterangan','Status'])
    st.dataframe(cols_info, use_container_width=True, hide_index=True)

    st.info("💡 **Separator desimal koma** (format Excel Indonesia) **otomatis dikonversi** saat upload — tidak perlu ubah manual.")

# ============================================================
# LOCAL FILE SCANNER
# ============================================================
CANDIDATE_FILES = [
    'dataset_absensi_final2.xlsx',
    'absen_pegawai.xlsx',
    'absensi.xlsx',
    'absensi_processed.xlsx',
    'absensi_processed.csv',
    'absensi.csv',
    'data_absensi.xlsx',
    'data_absensi.csv',
]

def scan_local_files():
    """Cari file dataset di folder yang sama dengan script."""
    import os
    found = []
    # Cek CANDIDATE_FILES dulu (prioritas)
    for fname in CANDIDATE_FILES:
        if os.path.exists(fname):
            found.append(fname)
    # Tambah semua xlsx/csv lain yang ada di folder
    for fname in sorted(os.listdir('.')):
        if fname.endswith(('.xlsx','.csv')) and fname not in found:
            found.append(fname)
    return found

@st.cache_data(show_spinner=False)
def load_local_file(filepath: str):
    """Load file lokal dari path, dengan cache."""
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
    fname = filepath.split('/')[-1].split('\\')[-1]
    return load_processed_file(file_bytes, fname)

# ============================================================
# PAGE: UPLOAD
# ============================================================
def page_upload(uploaded):
    st.markdown("## 📥 Upload Data Absensi")

    # ── DETEKSI FILE LOKAL ────────────────────────────────────
    local_files = scan_local_files()
    file_bytes  = None
    file_name   = None

    if local_files:
        st.success(f"📂 **{len(local_files)} file ditemukan** di folder yang sama dengan app ini.")
        col_sel, col_load = st.columns([4, 1])
        with col_sel:
            chosen_local = st.selectbox(
                "Pilih file lokal",
                options=local_files,
                key='local_file_select'
            )
        with col_load:
            st.markdown("<br>", unsafe_allow_html=True)
            load_local = st.button("📂 Load", type="primary", use_container_width=True,
                                   key='btn_load_local')
        if load_local:
            with st.spinner(f"⏳ Memuat {chosen_local}..."):
                df, fixed_cols = load_local_file(chosen_local)
            _finalize_load(df, fixed_cols, chosen_local)
            return

        st.markdown("---")

    # ── UPLOAD MANUAL ─────────────────────────────────────────
    if uploaded is None:
        if not local_files:
            st.warning("⬆️ Tidak ada file ditemukan di folder ini. Silakan upload melalui sidebar.")
        else:
            st.info("💡 Atau upload file lain dari komputer kamu:")
        return

    file_bytes = uploaded.getvalue()
    with st.spinner("⏳ Memuat dan memvalidasi data..."):
        df, fixed_cols = load_processed_file(file_bytes, uploaded.name)
    _finalize_load(df, fixed_cols, uploaded.name)

def _finalize_load(df, fixed_cols, source_name):
    """Tampilkan info & konfirmasi penggunaan data."""

    st.success(f"✅ **{source_name}** berhasil dimuat: **{len(df):,} baris**, **{len(df.columns)} kolom**")

    if fixed_cols:
        st.info(f"🔧 Auto-fix separator desimal pada kolom: `{'`, `'.join(fixed_cols)}`")

    warns = validate_dataframe(df)
    for w in warns:
        st.warning(f"⚠️ {w}")

    # Profiling
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Baris", f"{len(df):,}")
    with col2: st.metric("Karyawan Unik", f"{df['karyawan_id'].nunique():,}")
    with col3: st.metric("SKPD", f"{df['id_skpd'].nunique():,}" if 'id_skpd' in df.columns else '-')
    with col4:
        if 'risk_level' in df.columns:
            n_high = (df['risk_level']=='HIGH').sum()
            st.metric("HIGH Risk", f"{n_high:,} ({n_high/len(df)*100:.1f}%)")

    # Approver overview
    if 'approver_status' in df.columns:
        st.markdown("#### 📋 Overview Approver Status")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("✅ TERIMA", f"{df['is_terima'].sum():,}")
        with c2: st.metric("❌ TOLAK",  f"{df['is_tolak'].sum():,}")
        with c3: st.metric("⏳ PENDING", f"{df['is_pending'].sum():,}")
        with c4:
            fn = df['false_negative'].sum()
            st.metric("🚨 False Negative", f"{fn:,}",
                      help="TERIMA oleh approver tapi sistem menilai HIGH risk")

    with st.expander("🔍 Sample Data (10 baris pertama)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📊 Info Kolom", expanded=False):
        dtype_df = pd.DataFrame({
            'Kolom': df.dtypes.index,
            'Tipe': df.dtypes.values.astype(str),
            'Null': df.isnull().sum().values,
            'Null%': (df.isnull().sum().values/len(df)*100).round(1)
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    if st.button("✅ Gunakan Data Ini", type="primary", use_container_width=True):
        st.session_state['df'] = df
        st.session_state['office_centroid'] = build_office_centroid(df)
        st.session_state['file_name'] = uploaded.name
        st.success("✅ Data siap digunakan! Pilih menu **📊 Visualisasi** atau **🎯 Hunting**.")
        st.rerun()

# ============================================================
# PETA FOLIUM HELPER
# ============================================================
def build_popup(row):
    risk  = row.get('risk_level','N/A')
    color = risk_color_hex(risk)
    dist  = row.get('dist_km', 0)
    score = row.get('anomaly_score', 0)
    app_s = row.get('approver_status', '-') or '-'
    cat   = row.get('catatan', '-') or '-'
    fn    = row.get('false_negative', 0)
    fn_html = "<tr><td colspan=2><b style='color:#e74c3c'>⚠️ FALSE NEGATIVE — Approver kecolongan!</b></td></tr>" if fn else ""
    return f"""
    <div style='font-family:Arial;font-size:12px;min-width:240px'>
      <h4 style='margin:0 0 8px;color:#2c3e50'>📋 Detail Absensi</h4>
      <table style='width:100%;border-collapse:collapse'>
        <tr><td><b>Karyawan</b></td><td>{row.get('karyawan_id','')}</td></tr>
        <tr><td><b>SKPD</b></td><td>{row.get('id_skpd','')}</td></tr>
        <tr><td><b>Jenis</b></td><td>{'🟢 Masuk' if row.get('jenis')=='M' else '🔴 Pulang'}</td></tr>
        <tr><td><b>Waktu</b></td><td>{str(row.get('tanggal_kirim',''))[:16]}</td></tr>
        <tr><td><b>Jarak ke kantor</b></td><td>{dist:.3f} km</td></tr>
        <tr><td><b>Anomaly Score</b></td><td>{score}</td></tr>
        <tr><td><b>Risk Level</b></td><td><b style='color:{color}'>{risk}</b></td></tr>
        <tr><td><b>Approver</b></td><td>{app_s}</td></tr>
        <tr><td><b>Catatan</b></td><td>{cat}</td></tr>
        {fn_html}
      </table>
    </div>"""

def create_folium_map(df, map_type='marker', office_centroid=None):
    center_lat  = df['lat'].median()
    center_long = df['long'].median()
    m = folium.Map(location=[center_lat, center_long], zoom_start=13, tiles='CartoDB positron')
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)

    if map_type == 'heatmap':
        heat = [[r['lat'], r['long'], r.get('anomaly_score',1)+1] for _, r in df.iterrows()]
        HeatMap(heat, radius=15, blur=10).add_to(m)
    elif map_type == 'cluster':
        mc = MarkerCluster(name='Absensi').add_to(m)
        for _, row in df.iterrows():
            c = risk_color_folium(row.get('risk_level','LOW'))
            folium.CircleMarker(
                location=[row['lat'], row['long']], radius=7,
                color=c, fill=True, fill_color=c, fill_opacity=0.75,
                popup=folium.Popup(build_popup(row), max_width=280)
            ).add_to(mc)
    else:
        for risk in ['HIGH','MED','LOW']:
            fg = folium.FeatureGroup(name=f'Risk {risk}')
            c  = risk_color_folium(risk)
            for _, row in df[df['risk_level']==risk].iterrows():
                # False negative = border kuning
                ec = 'gold' if row.get('false_negative',0) == 1 else c
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=9 if risk=='HIGH' else 6,
                    color=ec, fill=True, fill_color=c, fill_opacity=0.75,
                    popup=folium.Popup(build_popup(row), max_width=280)
                ).add_to(fg)
            fg.add_to(m)

    if office_centroid is not None and len(office_centroid) > 0:
        fg_off = folium.FeatureGroup(name='Kantor SKPD')
        for _, o in office_centroid.iterrows():
            if pd.notna(o.get('office_lat')):
                folium.Marker(
                    [o['office_lat'], o['office_long']],
                    popup=f"Kantor SKPD {o['id_skpd']}",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa')
                ).add_to(fg_off)
                folium.Circle([o['office_lat'], o['office_long']],
                              radius=300, color='#3498db', fill=False,
                              weight=2, dash_array='5').add_to(fg_off)
        fg_off.add_to(m)

    folium.LayerControl().add_to(m)
    return m

# ============================================================
# PAGE: VISUALISASI
# ============================================================
def page_visualisasi(filters):
    st.markdown("## 📊 Visualisasi Anomali Absensi")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Belum ada data. Upload dulu di menu **📥 Upload Data**.")
        return

    df_full = st.session_state.df
    oc      = st.session_state.get('office_centroid')

    h = _df_hash(df_full)
    df = apply_filters(
        h, df_full,
        filters.get('skpd','Semua'),
        tuple(filters.get('risk',['HIGH','MED','LOW'])),
        tuple(filters.get('jenis',['M','P'])),
        filters.get('date'),
        filters.get('dist',(0.0,100.0)),
        filters.get('approver','Semua')
    )
    st.caption(f"📊 Menampilkan **{len(df):,}** dari **{len(df_full):,}** absensi")

    tabs = st.tabs(["📊 Overview","🗺️ Peta","⏰ Temporal","📏 Jarak","👤 Karyawan","📋 Approver","📋 Data"])
    with tabs[0]: _vis_overview(df)
    with tabs[1]: _vis_map(df, filters, oc)
    with tabs[2]: _vis_temporal(df)
    with tabs[3]: _vis_distance(df)
    with tabs[4]: _vis_employee(df)
    with tabs[5]: _vis_approver(df)
    with tabs[6]: _vis_data(df)

def _vis_overview(df):
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Total Absensi",  f"{len(df):,}")
    with c2: st.metric("Karyawan",       f"{df['karyawan_id'].nunique():,}")
    with c3:
        h = (df['risk_level']=='HIGH').sum()
        st.metric("🔴 HIGH", f"{h:,}", f"{h/len(df)*100:.1f}%" if len(df) else "0%")
    with c4: st.metric("🟡 MED", f"{(df['risk_level']=='MED').sum():,}")
    with c5: st.metric("🟢 LOW", f"{(df['risk_level']=='LOW').sum():,}")

    if 'false_negative' in df.columns and df['false_negative'].sum() > 0:
        fn = df['false_negative'].sum()
        st.markdown(f"""<div class='alert-box alert-red'>
        🚨 <b>Peringatan False Negative:</b> {fn:,} absensi dinilai <b>HIGH risk</b> oleh sistem
        tapi sudah di-<b>TERIMA</b> oleh approver — perlu review manual!
        </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.pie(df['risk_level'].value_counts().reset_index(),
                     values='count', names='risk_level', title='Distribusi Risk Level',
                     color='risk_level',
                     color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'}, hole=0.4)
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        skpd_risk = df.groupby(['id_skpd','risk_level']).size().reset_index(name='count')
        fig = px.bar(skpd_risk, x='id_skpd', y='count', color='risk_level',
                     title='Distribusi Risk per SKPD', barmode='stack',
                     color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x='anomaly_score', color='risk_level', nbins=30,
                       title='Distribusi Anomaly Score',
                       color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)

def _vis_map(df, filters, oc):
    if df.empty:
        st.warning("Tidak ada data.")
        return
    MAX = 2000
    df_d = df.sample(MAX, random_state=42) if len(df) > MAX else df
    if len(df) > MAX:
        st.info(f"Menampilkan {MAX:,} dari {len(df):,} titik.")
    m = create_folium_map(df_d, filters.get('map_type','marker'), oc)
    st_folium(m, width=None, height=560, returned_objects=[])
    st.markdown("🔴 HIGH &nbsp; 🟡 MED &nbsp; 🟢 LOW &nbsp; 🔵 Kantor &nbsp; ⭕ 300m &nbsp; 🟡-border = False Negative")

def _vis_temporal(df):
    if 'jam' in df.columns:
        col1,col2 = st.columns(2)
        with col1:
            fig = px.bar(df.groupby(['jam','risk_level']).size().reset_index(name='n'),
                         x='jam', y='n', color='risk_level', title='Per Jam',
                         color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
            fig.add_vrect(x0=7,x1=9, fillcolor='green', opacity=0.07, annotation_text='Jam Masuk')
            fig.add_vrect(x0=15,x1=17, fillcolor='purple', opacity=0.07, annotation_text='Jam Pulang')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'weekday' in df.columns:
                dm = {0:'Senin',1:'Selasa',2:'Rabu',3:'Kamis',4:'Jumat',5:'Sabtu',6:'Minggu'}
                df2 = df.copy(); df2['hari'] = df2['weekday'].map(dm)
                fig = px.bar(df2.groupby(['hari','risk_level']).size().reset_index(name='n'),
                             x='hari', y='n', color='risk_level', title='Per Hari',
                             color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'},
                             category_orders={'hari':list(dm.values())})
                st.plotly_chart(fig, use_container_width=True)

    if 'tanggal' in df.columns:
        daily = df.groupby(['tanggal','risk_level']).size().reset_index(name='n')
        fig = px.line(daily, x='tanggal', y='n', color='risk_level', markers=True,
                      title='Trend Harian',
                      color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

def _vis_distance(df):
    if 'dist_km' not in df.columns:
        st.warning("Kolom dist_km tidak ditemukan."); return
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Rata-rata", f"{df['dist_km'].mean():.3f} km")
    with c2: st.metric("Median",    f"{df['dist_km'].median():.3f} km")
    with c3: st.metric("Maksimum",  f"{df['dist_km'].max():.3f} km")
    with c4:
        out = (df['dist_km']>0.3).sum()
        st.metric("Di luar 300m", f"{out:,} ({out/len(df)*100:.1f}%)")

    col_l,col_r = st.columns(2)
    with col_l:
        fig = px.histogram(df[df['dist_km']<=10], x='dist_km', color='risk_level',
                           title='Distribusi Jarak ≤10km', nbins=50,
                           color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
        fig.add_vline(x=0.3, line_dash='dash', line_color='red', annotation_text='300m')
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.box(df[df['dist_km']<=10], x='id_skpd', y='dist_km', color='risk_level',
                     title='Boxplot Jarak per SKPD',
                     color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
        st.plotly_chart(fig, use_container_width=True)

def _vis_employee(df):
    agg = df.groupby('karyawan_id').agg(
        total=('karyawan_id','count'),
        high_count=('risk_level', lambda x: (x=='HIGH').sum()),
        avg_score=('anomaly_score','mean'),
        max_dist=('dist_km','max'),
        skpd=('id_skpd','first')
    ).reset_index()
    agg['high_pct'] = (agg['high_count']/agg['total']*100).round(1)

    col_l,col_r = st.columns(2)
    with col_l:
        top = agg.nlargest(10,'high_count')
        fig = px.bar(top, x='karyawan_id', y='high_count', title='Top 10 HIGH Risk',
                     color='high_pct', color_continuous_scale='Reds')
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.scatter(agg, x='total', y='avg_score', size='max_dist',
                         color='high_pct', hover_data=['karyawan_id','skpd'],
                         title='Total Absensi vs Avg Score',
                         color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)

    risky = agg[agg['high_count']>0].sort_values('high_count', ascending=False)
    if len(risky):
        st.markdown("### 🚨 Karyawan Berisiko Tinggi")
        st.dataframe(risky.head(30), use_container_width=True)

def _vis_approver(df):
    st.markdown("### 📋 Analisis Approver Status")
    if 'approver_status' not in df.columns:
        st.info("Kolom approver_status tidak ada di dataset ini."); return

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("✅ TERIMA",        f"{df['is_terima'].sum():,}")
    with c2: st.metric("❌ TOLAK",         f"{df['is_tolak'].sum():,}")
    with c3: st.metric("⏳ PENDING",       f"{df['is_pending'].sum():,}")
    with c4: st.metric("🚨 False Negative",f"{df['false_negative'].sum():,}")

    col_l,col_r = st.columns(2)
    with col_l:
        # Approver vs Risk cross-tab
        cross = df.groupby(['risk_level','approver_status']).size().reset_index(name='n')
        cross = cross[cross['approver_status'] != '']
        fig = px.bar(cross, x='risk_level', y='n', color='approver_status',
                     title='Risk Level vs Keputusan Approver', barmode='group')
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        # TOLAK per SKPD
        tolak_skpd = df[df['is_tolak']==1].groupby('id_skpd').size().reset_index(name='n_tolak')
        fig = px.bar(tolak_skpd.sort_values('n_tolak',ascending=False),
                     x='id_skpd', y='n_tolak', title='TOLAK per SKPD',
                     color='n_tolak', color_continuous_scale='Reds')
        fig.update_xaxes(type='category')
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    # False Negative detail
    fn_df = df[df['false_negative']==1].sort_values('anomaly_score', ascending=False)
    if len(fn_df):
        st.markdown(f"""<div class='alert-box alert-red'>
        🚨 <b>{len(fn_df):,} False Negative</b> — absensi HIGH risk yang sudah di-TERIMA approver.
        Butuh review manual segera!
        </div>""", unsafe_allow_html=True)
        cols = ['karyawan_id','id_skpd','jenis','tanggal_kirim','dist_km',
                'anomaly_score','risk_level','approver_status','catatan']
        cols = [c for c in cols if c in fn_df.columns]
        st.dataframe(fn_df[cols].head(50), use_container_width=True)
        csv = fn_df[cols].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download False Negative CSV", csv,
                           "false_negative.csv", "text/csv")

    # PENDING lama
    pending_df = df[df['is_pending']==1]
    if len(pending_df):
        st.markdown(f"""<div class='alert-box alert-orange'>
        ⏳ <b>{len(pending_df):,} absensi masih PENDING</b> belum ada keputusan approver.
        </div>""", unsafe_allow_html=True)
        pend_agg = pending_df.groupby(['karyawan_id','id_skpd','risk_level']).size().reset_index(name='n_pending')
        pend_agg = pend_agg.sort_values('n_pending', ascending=False)
        st.dataframe(pend_agg.head(30), use_container_width=True)

def _vis_data(df):
    col1,col2 = st.columns(2)
    with col1: search = st.text_input("🔍 Cari Karyawan ID","")
    with col2: sort_col = st.selectbox("Urutkan", ['anomaly_score','dist_km','tanggal_kirim'])
    df_t = df.copy()
    if search:
        df_t = df_t[df_t['karyawan_id'].astype(str).str.contains(search)]
    if sort_col in df_t.columns:
        df_t = df_t.sort_values(sort_col, ascending=False)
    cols_show = ['karyawan_id','id_skpd','jenis','tanggal_kirim','lat','long',
                 'dist_km','anomaly_score','risk_level','approver_status',
                 'false_negative','system_action','outside_300m','very_far',
                 'extreme_far','far_no_note','catatan']
    cols_show = [c for c in cols_show if c in df_t.columns]
    st.dataframe(df_t[cols_show].head(500), use_container_width=True, height=480)
    st.caption(f"Menampilkan {min(500,len(df_t))} dari {len(df_t):,} baris")
    csv = df_t[cols_show].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "absensi_filtered.csv", "text/csv")

# ============================================================
# PAGE: HUNTING
# ============================================================
def page_hunting():
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Belum ada data. Upload dulu di menu **📥 Upload Data**."); return

    df  = st.session_state.df
    oc  = st.session_state.get('office_centroid', pd.DataFrame())

    st.markdown("""
    <div class="hunt-header">
        <div class="hunt-title">[ HUNTING MODE ]</div>
        <div class="hunt-sub">Investigasi mendalam anomali absensi — drill down by pegawai, unit kerja, atau tanggal</div>
    </div>""", unsafe_allow_html=True)

    total = len(df)
    nh = (df['risk_level']=='HIGH').sum()
    nm = (df['risk_level']=='MED').sum()
    nl = (df['risk_level']=='LOW').sum()
    fn = df.get('false_negative', pd.Series([0]*len(df))).sum() if 'false_negative' in df.columns else 0

    st.markdown(f"""
    <div style="background:#f0f2f6;border-radius:8px;padding:.6rem 1.2rem;margin-bottom:1rem;
                display:flex;gap:2rem;align-items:center;font-size:.86rem;flex-wrap:wrap">
        <span>📊 <b>{total:,}</b> absensi</span>
        <span>🔴 <b style="color:#e74c3c">{nh:,}</b> HIGH</span>
        <span>🟡 <b style="color:#f39c12">{nm:,}</b> MED</span>
        <span>🟢 <b style="color:#27ae60">{nl:,}</b> LOW</span>
        <span>👤 <b>{df['karyawan_id'].nunique():,}</b> karyawan</span>
        <span>🏢 <b>{df['id_skpd'].nunique():,}</b> SKPD</span>
        {"<span>🚨 <b style='color:#e74c3c'>" + str(fn) + "</b> False Negative</span>" if fn > 0 else ""}
    </div>""", unsafe_allow_html=True)

    tab_peg, tab_skpd, tab_tgl = st.tabs([
        "🕵️ Section 1 — By Pegawai",
        "🏢 Section 2 — By SKPD",
        "📅 Section 3 — By Tanggal",
    ])
    with tab_peg:  _hunt_pegawai(df, oc)
    with tab_skpd: _hunt_skpd(df, oc)
    with tab_tgl:  _hunt_tanggal(df, oc)

# ── HUNT: PEGAWAI ────────────────────────────────────────────
def _hunt_pegawai(df, oc):
    st.markdown("""<div class="section-header">
        <span class="section-icon">🕵️</span>
        <div><div class="section-title">Hunt by Pegawai</div>
        <div class="section-desc">Timeline, jejak lokasi, perbandingan SKPD, tren risiko, analisis approver</div></div>
    </div>""", unsafe_allow_html=True)

    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = []

    all_ids = sorted(df['karyawan_id'].unique().tolist())
    col_s, col_b = st.columns([4,1])
    with col_s:
        sel_id = st.selectbox(
            "🔎 Pilih ID Pegawai",
            all_ids,
            format_func=lambda x: (
                f"ID {x} | SKPD {df[df['karyawan_id']==x]['id_skpd'].iloc[0] if len(df[df['karyawan_id']==x])>0 else '-'}"
                f" | HIGH: {(df[df['karyawan_id']==x]['risk_level']=='HIGH').sum()}"
                f" | Score avg: {df[df['karyawan_id']==x]['anomaly_score'].mean():.0f}"
            ),
            key='hunt_peg_id'
        )
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        in_wl = sel_id in st.session_state['watchlist']
        if st.button("📌 Watchlist" if not in_wl else "❌ Hapus WL", use_container_width=True):
            if in_wl: st.session_state['watchlist'].remove(sel_id)
            else:      st.session_state['watchlist'].append(sel_id)
            st.rerun()

    df_e = df[df['karyawan_id'] == sel_id].sort_values('tanggal_kirim')
    if df_e.empty:
        st.warning("Tidak ada data."); return

    total_e = len(df_e)
    n_high  = (df_e['risk_level']=='HIGH').sum()
    n_med   = (df_e['risk_level']=='MED').sum()
    avg_km  = df_e['dist_km'].mean()
    max_km  = df_e['dist_km'].max()
    avg_sc  = df_e['anomaly_score'].mean()
    skpd_e  = df_e['id_skpd'].mode()[0]

    # Approver summary untuk karyawan ini
    n_tolak  = df_e.get('is_tolak', pd.Series([0]*len(df_e))).sum() if 'is_tolak' in df_e.columns else 0
    n_fn     = df_e.get('false_negative', pd.Series([0]*len(df_e))).sum() if 'false_negative' in df_e.columns else 0
    n_pend   = df_e.get('is_pending', pd.Series([0]*len(df_e))).sum() if 'is_pending' in df_e.columns else 0

    # Trend
    if 'tanggal_kirim' in df_e.columns and len(df_e) >= 4:
        last_d  = df_e['tanggal_kirim'].max()
        cutoff  = last_d - timedelta(days=7)
        recent  = df_e[df_e['tanggal_kirim'] >= cutoff]['anomaly_score'].mean()
        older   = df_e[df_e['tanggal_kirim'] <  cutoff]['anomaly_score'].mean()
        if pd.notna(recent) and pd.notna(older) and older > 0:
            pct = (recent-older)/older*100
            trend_html = (f"<span class='trend-up'>▲ {pct:.0f}% Memburuk</span>" if pct > 10
                          else f"<span class='trend-down'>▼ {abs(pct):.0f}% Membaik</span>" if pct < -10
                          else "<span class='trend-flat'>➔ Stabil</span>")
        else:
            trend_html = "<span class='trend-flat'>Data terbatas</span>"
    else:
        trend_html = ""

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card info"><div class="metric-val">{total_e}</div><div class="metric-lbl">Total Absensi</div></div>
        <div class="metric-card danger"><div class="metric-val">{n_high}</div><div class="metric-lbl">HIGH Risk</div></div>
        <div class="metric-card warning"><div class="metric-val">{n_med}</div><div class="metric-lbl">MED Risk</div></div>
        <div class="metric-card"><div class="metric-val">{avg_km:.3f} km</div><div class="metric-lbl">Avg Jarak</div></div>
        <div class="metric-card danger"><div class="metric-val">{max_km:.3f} km</div><div class="metric-lbl">Max Jarak</div></div>
        <div class="metric-card"><div class="metric-val">{avg_sc:.0f}</div><div class="metric-lbl">Avg Score</div></div>
        <div class="metric-card {'danger' if n_fn>0 else ''}"><div class="metric-val">{n_fn}</div><div class="metric-lbl">False Neg.</div></div>
        <div class="metric-card {'warning' if n_tolak>0 else ''}"><div class="metric-val">{n_tolak}</div><div class="metric-lbl">TOLAK</div></div>
        <div class="metric-card"><div class="metric-val">{n_pend}</div><div class="metric-lbl">Pending</div></div>
    </div>
    {"<div style='font-size:.84rem;margin-bottom:.8rem'>📈 Tren 7 hari terakhir: " + trend_html + "</div>" if trend_html else ""}
    """, unsafe_allow_html=True)

    if n_fn > 0:
        st.markdown(f"""<div class='alert-box alert-red'>
        🚨 <b>{n_fn} False Negative</b> — karyawan ini punya absensi HIGH risk yang sudah di-TERIMA approver!
        </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6 = st.tabs(["📅 Timeline","🗺️ Jejak Lokasi","📊 Vs SKPD","📈 Tren","📋 Approver","📋 Riwayat"])

    with t1:
        st.markdown("""<div class='timeline-tip'>
        💡 Titik besar + merah = HIGH risk. Perhatikan jam & pola hari — ada hari tertentu yang selalu anomali?
        </div>""", unsafe_allow_html=True)
        if 'tanggal' in df_e.columns and 'jam_desimal' in df_e.columns:
            hover = ['dist_km','anomaly_score']
            if 'catatan' in df_e.columns: hover.append('catatan')
            if 'approver_status' in df_e.columns: hover.append('approver_status')
            fig = px.scatter(df_e, x='tanggal', y='jam_desimal',
                             color='risk_level', size='anomaly_score', symbol='jenis',
                             color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'},
                             title=f'Timeline — Karyawan {sel_id}', hover_data=hover,
                             labels={'tanggal':'Tanggal','jam_desimal':'Jam'})
            fig.add_hline(y=7.5,  line_dash='dot', line_color='#3498db', annotation_text='07:30')
            fig.add_hline(y=16.0, line_dash='dot', line_color='#9b59b6', annotation_text='16:00')
            fig.update_layout(height=420, plot_bgcolor='#fafafa')
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        ctr = [df_e['lat'].median(), df_e['long'].median()]
        m = folium.Map(location=ctr, zoom_start=14, tiles='CartoDB positron')
        coords = [[r['lat'],r['long']] for _,r in df_e.iterrows()]
        if len(coords) > 1:
            AntPath(locations=coords, color='#667eea', weight=2.5,
                    opacity=0.6, delay=800, dash_array=[10,20]).add_to(m)
        for i,(_,row) in enumerate(df_e.iterrows()):
            c = risk_color_folium(row.get('risk_level','LOW'))
            border = 'gold' if row.get('false_negative',0)==1 else c
            popup = f"""<div style='font-size:12px;min-width:190px'>
                <b>#{i+1} — {str(row.get('tanggal_kirim',''))[:16]}</b><br>
                {'🟢 Masuk' if row.get('jenis')=='M' else '🔴 Pulang'}<br>
                Jarak: {row.get('dist_km',0):.3f} km | Score: {row.get('anomaly_score',0)}<br>
                Risk: <b>{row.get('risk_level','')}</b><br>
                Approver: {row.get('approver_status','-') or '-'}<br>
                {'⚠️ FALSE NEGATIVE' if row.get('false_negative',0)==1 else ''}
            </div>"""
            folium.CircleMarker(
                location=[row['lat'],row['long']],
                radius=11 if row.get('risk_level')=='HIGH' else 7,
                color=border, fill=True, fill_color=c, fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=230),
                tooltip=f"#{i+1}|{row.get('risk_level','')}"
            ).add_to(m)
        # Kantor
        if not oc.empty:
            off = oc[oc['id_skpd']==skpd_e]
            if not off.empty:
                o = off.iloc[0]
                folium.Marker([o['office_lat'],o['office_long']],
                              popup=f"Kantor SKPD {skpd_e}",
                              icon=folium.Icon(color='blue',icon='home',prefix='fa')).add_to(m)
                folium.Circle([o['office_lat'],o['office_long']], radius=300,
                              color='#3498db', fill=False, weight=2, dash_array='5').add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=500, returned_objects=[])
        st.caption("🟡-border = False Negative | Garis animasi = urutan absensi")

    with t3:
        df_skpd_all = df[df['id_skpd']==skpd_e]
        agg_skpd = df_skpd_all.groupby('karyawan_id').agg(
            avg_score=('anomaly_score','mean'),
            avg_dist=('dist_km','mean'),
            high_pct=('risk_level', lambda x: (x=='HIGH').mean()*100)
        ).reset_index()
        emp_row = agg_skpd[agg_skpd['karyawan_id']==sel_id]
        if not emp_row.empty:
            e = emp_row.iloc[0]
            def rank_str(series, val, higher_is_bad=True):
                rank = (series>val).sum()+1 if higher_is_bad else (series<val).sum()+1
                total = len(series)
                pct = rank/total*100
                ico = '🔴' if pct<30 else '🟡' if pct<60 else '🟢'
                return f"{ico} Peringkat **{rank}** dari {total} karyawan"
            st.markdown(f"""
            #### Posisi di SKPD {skpd_e}
            - Avg Score: {e['avg_score']:.1f} — {rank_str(agg_skpd['avg_score'], e['avg_score'])}
            - Avg Jarak: {e['avg_dist']:.3f} km — {rank_str(agg_skpd['avg_dist'], e['avg_dist'])}
            - % HIGH: {e['high_pct']:.1f}% — {rank_str(agg_skpd['high_pct'], e['high_pct'])}
            """)
        fig = px.scatter(agg_skpd, x='avg_dist', y='avg_score', size='high_pct',
                         color='high_pct', color_continuous_scale='RdYlGn_r',
                         hover_data=['karyawan_id'],
                         title=f'Sebaran Karyawan SKPD {skpd_e}')
        if not emp_row.empty:
            e = emp_row.iloc[0]
            fig.add_annotation(x=e['avg_dist'], y=e['avg_score'],
                               text=f"▶ ID {sel_id}", showarrow=True,
                               arrowhead=2, font=dict(color='#e74c3c',size=12))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        if 'tanggal' in df_e.columns:
            daily = df_e.groupby('tanggal').agg(
                avg_score=('anomaly_score','mean'),
                max_score=('anomaly_score','max'),
                n_high=('risk_level', lambda x: (x=='HIGH').sum())
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily['tanggal'], y=daily['avg_score'],
                                     name='Avg Score', line=dict(color='#3498db',width=2), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=daily['tanggal'], y=daily['max_score'],
                                     name='Max Score', line=dict(color='#e74c3c',width=1.5,dash='dash'), mode='lines+markers'))
            fig.add_trace(go.Bar(x=daily['tanggal'], y=daily['n_high'],
                                 name='# HIGH', marker_color='rgba(231,76,60,.2)', yaxis='y2'))
            fig.update_layout(
                title=f'Tren Harian Karyawan {sel_id}',
                yaxis=dict(title='Score'), height=400,
                yaxis2=dict(title='# HIGH', overlaying='y', side='right'),
                hovermode='x unified'
            )
            fig.add_hrect(y0=70, y1=max(daily['max_score'].max()+5, 75),
                          fillcolor='red', opacity=0.04, line_width=0)
            fig.add_hrect(y0=30, y1=70, fillcolor='orange', opacity=0.04, line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    with t5:
        if 'approver_status' in df_e.columns:
            st.markdown("#### Riwayat Approver Karyawan Ini")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("TERIMA", f"{df_e['is_terima'].sum()}")
            with c2: st.metric("TOLAK",  f"{df_e['is_tolak'].sum()}")
            with c3: st.metric("PENDING",f"{df_e['is_pending'].sum()}")

            # Cross risk vs approver
            cross = df_e.groupby(['risk_level','approver_status']).size().reset_index(name='n')
            cross = cross[cross['approver_status'] != '']
            if not cross.empty:
                fig = px.bar(cross, x='risk_level', y='n', color='approver_status',
                             title='Risk vs Keputusan Approver', barmode='group')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            fn_rows = df_e[df_e['false_negative']==1]
            if len(fn_rows):
                st.error(f"🚨 {len(fn_rows)} False Negative — HIGH risk yang di-TERIMA approver!")
                cols = ['tanggal_kirim','jenis','dist_km','anomaly_score','approver_status','catatan']
                cols = [c for c in cols if c in fn_rows.columns]
                st.dataframe(fn_rows[cols], use_container_width=True)

    with t6:
        cols_show = ['tanggal_kirim','jenis','lat','long','dist_km','anomaly_score',
                     'risk_level','approver_status','false_negative',
                     'outside_300m','very_far','extreme_far','far_no_note','catatan']
        cols_show = [c for c in cols_show if c in df_e.columns]
        st.dataframe(df_e[cols_show].sort_values('tanggal_kirim', ascending=False),
                     use_container_width=True, height=400)
        csv = df_e[cols_show].to_csv(index=False).encode('utf-8')
        st.download_button(f"⬇️ Download Riwayat ID {sel_id}", csv,
                           f"karyawan_{sel_id}.csv", "text/csv")

# ── HUNT: SKPD ───────────────────────────────────────────────
def _hunt_skpd(df, oc):
    st.markdown("""<div class="section-header">
        <span class="section-icon">🏢</span>
        <div><div class="section-title">Hunt by SKPD</div>
        <div class="section-desc">Leaderboard, heatmap, anomali kolektif, radar benchmark, analisis approver</div></div>
    </div>""", unsafe_allow_html=True)

    skpd_list = sorted(df['id_skpd'].unique().tolist())
    sel_skpd  = st.selectbox("🏢 Pilih SKPD",
                              skpd_list,
                              format_func=lambda x: f"SKPD {x} ({len(df[df['id_skpd']==x]):,} absensi)",
                              key='hunt_skpd_id')
    df_s = df[df['id_skpd']==sel_skpd].copy()
    if df_s.empty:
        st.warning("Tidak ada data."); return

    n_kar = df_s['karyawan_id'].nunique()
    n_fn  = df_s['false_negative'].sum() if 'false_negative' in df_s.columns else 0
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card info"><div class="metric-val">{len(df_s):,}</div><div class="metric-lbl">Total Absensi</div></div>
        <div class="metric-card info"><div class="metric-val">{n_kar}</div><div class="metric-lbl">Karyawan</div></div>
        <div class="metric-card danger"><div class="metric-val">{(df_s['risk_level']=='HIGH').sum():,}</div><div class="metric-lbl">HIGH Risk</div></div>
        <div class="metric-card"><div class="metric-val">{(df_s['risk_level']=='HIGH').sum()/len(df_s)*100:.1f}%</div><div class="metric-lbl">% HIGH</div></div>
        <div class="metric-card"><div class="metric-val">{df_s['dist_km'].mean():.3f} km</div><div class="metric-lbl">Avg Jarak</div></div>
        <div class="metric-card {'danger' if n_fn>0 else ''}"><div class="metric-val">{n_fn}</div><div class="metric-lbl">False Neg.</div></div>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6 = st.tabs([
        "🏆 Leaderboard","🔥 Heatmap","📏 Distribusi Jarak",
        "📅 Anomali Kolektif","📡 Radar","📋 Approver SKPD"
    ])

    with t1:
        agg = df_s.groupby('karyawan_id').agg(
            total=('karyawan_id','count'),
            high_count=('risk_level', lambda x: (x=='HIGH').sum()),
            med_count=('risk_level',  lambda x: (x=='MED').sum()),
            avg_score=('anomaly_score','mean'),
            max_dist=('dist_km','max'),
            n_tolak=('is_tolak','sum') if 'is_tolak' in df_s.columns else ('karyawan_id', lambda x: 0),
            n_fn=('false_negative','sum') if 'false_negative' in df_s.columns else ('karyawan_id', lambda x: 0),
        ).reset_index()
        agg['high_pct'] = (agg['high_count']/agg['total']*100).round(1)
        agg = agg.sort_values('high_count', ascending=False)

        fig = px.bar(agg.head(15), x='karyawan_id', y=['high_count','med_count'],
                     title=f'Leaderboard Anomali SKPD {sel_skpd}',
                     color_discrete_map={'high_count':'#e74c3c','med_count':'#f39c12'},
                     barmode='stack')
        fig.update_xaxes(type='category')
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(agg, use_container_width=True, height=320)

    with t2:
        m = folium.Map(location=[df_s['lat'].median(), df_s['long'].median()],
                       zoom_start=13, tiles='CartoDB positron')
        heat = [[r['lat'],r['long'],r.get('anomaly_score',1)+1] for _,r in df_s.iterrows()]
        HeatMap(heat, radius=18, blur=12,
                gradient={'0.0':'green','0.5':'yellow','1.0':'red'}).add_to(m)
        if not oc.empty:
            off = oc[oc['id_skpd']==sel_skpd]
            if not off.empty:
                o = off.iloc[0]
                folium.Marker([o['office_lat'],o['office_long']],
                              popup=f"Kantor SKPD {sel_skpd}",
                              icon=folium.Icon(color='blue',icon='home',prefix='fa')).add_to(m)
                folium.Circle([o['office_lat'],o['office_long']], radius=300,
                              color='blue', fill=False, weight=2, dash_array='5').add_to(m)
        st_folium(m, width=None, height=500, returned_objects=[])

    with t3:
        fig = px.box(df_s[df_s['dist_km']<=15], x='karyawan_id', y='dist_km', color='risk_level',
                     title=f'Distribusi Jarak per Karyawan SKPD {sel_skpd}',
                     color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
        fig.add_hline(y=0.3, line_dash='dash', line_color='gray', annotation_text='300m')
        fig.update_xaxes(type='category')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        if 'tanggal' in df_s.columns:
            daily_col = df_s[df_s['outside_300m']==1].groupby('tanggal').agg(
                n_karyawan=('karyawan_id','nunique'),
                n_absensi=('karyawan_id','count'),
                avg_score=('anomaly_score','mean')
            ).reset_index().sort_values('n_karyawan', ascending=False)

            threshold = max(2, int(n_kar*0.3))
            massal = daily_col[daily_col['n_karyawan'] >= threshold]
            if len(massal):
                st.warning(f"⚠️ {len(massal)} hari — ≥{threshold} karyawan absen di luar kantor secara bersamaan!")
                fig = px.bar(massal, x='tanggal', y='n_karyawan',
                             color='avg_score', color_continuous_scale='Reds',
                             title='Hari Anomali Massal')
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(massal, use_container_width=True)
            else:
                st.success("✅ Tidak ada anomali kolektif massal.")

            daily_all = df_s.groupby(['tanggal','risk_level']).size().reset_index(name='n')
            fig2 = px.area(daily_all, x='tanggal', y='n', color='risk_level',
                           title='Trend Harian Semua Absensi SKPD',
                           color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

    with t5:
        all_agg = df.groupby('id_skpd').agg(
            avg_score=('anomaly_score','mean'),
            high_pct=('risk_level', lambda x: (x=='HIGH').mean()*100),
            avg_dist=('dist_km','mean'),
            far_pct=('outside_300m','mean'),
            extreme_pct=('extreme_far','mean')
        ).reset_index()
        cur = all_agg[all_agg['id_skpd']==sel_skpd]
        mean_v = all_agg.mean(numeric_only=True)
        if not cur.empty:
            c = cur.iloc[0]
            cats = ['Avg Score','% HIGH','Avg Jarak×10','% Luar 300m','% Extreme']
            vc = [c['avg_score'], c['high_pct'], c['avg_dist']*10, c['far_pct']*100, c['extreme_pct']*100]
            vm = [mean_v['avg_score'], mean_v['high_pct'], mean_v['avg_dist']*10, mean_v['far_pct']*100, mean_v['extreme_pct']*100]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=vc+[vc[0]], theta=cats+[cats[0]],
                                          fill='toself', name=f'SKPD {sel_skpd}',
                                          line=dict(color='#e74c3c',width=2)))
            fig.add_trace(go.Scatterpolar(r=vm+[vm[0]], theta=cats+[cats[0]],
                                          fill='toself', name='Rata-rata',
                                          line=dict(color='#3498db',width=2), opacity=0.5))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                              title=f'Radar SKPD {sel_skpd} vs Rata-rata', height=430)
            st.plotly_chart(fig, use_container_width=True)

    with t6:
        if 'approver_status' in df_s.columns:
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("TERIMA", f"{df_s['is_terima'].sum():,}")
            with c2: st.metric("TOLAK",  f"{df_s['is_tolak'].sum():,}")
            with c3: st.metric("PENDING",f"{df_s['is_pending'].sum():,}")
            with c4: st.metric("False Neg.", f"{n_fn:,}")

            cross = df_s.groupby(['risk_level','approver_status']).size().reset_index(name='n')
            cross = cross[cross['approver_status']!='']
            if not cross.empty:
                fig = px.bar(cross, x='risk_level', y='n', color='approver_status',
                             title='Risk vs Approver SKPD Ini', barmode='group')
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

            if n_fn > 0:
                fn_rows = df_s[df_s['false_negative']==1]
                st.error(f"🚨 {n_fn} False Negative di SKPD ini!")
                cols = ['karyawan_id','tanggal_kirim','dist_km','anomaly_score','approver_status']
                cols = [c for c in cols if c in fn_rows.columns]
                st.dataframe(fn_rows[cols], use_container_width=True)

# ── HUNT: TANGGAL ────────────────────────────────────────────
def _hunt_tanggal(df, oc):
    st.markdown("""<div class="section-header">
        <span class="section-icon">📅</span>
        <div><div class="section-title">Hunt by Tanggal</div>
        <div class="section-desc">Snapshot harian, distribusi jam, daftar karyawan, deteksi titipan absensi</div></div>
    </div>""", unsafe_allow_html=True)

    if 'tanggal' not in df.columns:
        st.warning("Kolom tanggal tidak tersedia."); return

    min_d = df['tanggal'].min(); max_d = df['tanggal'].max()
    col1,col2,col3 = st.columns([2,2,1])
    with col1:
        date_range = st.date_input("📅 Rentang Tanggal",
                                   value=(max_d - timedelta(days=6), max_d),
                                   min_value=min_d, max_value=max_d,
                                   key='hunt_date_range')
    with col2:
        filter_risk = st.multiselect("Filter Risk", ['HIGH','MED','LOW'],
                                     default=['HIGH','MED','LOW'], key='hunt_dt_risk')
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        approver_f = st.selectbox("Approver", ['Semua','TOLAK','False Negative','PENDING'],
                                  key='hunt_dt_app') if 'approver_status' in df.columns else 'Semua'

    d_start, d_end = (date_range if isinstance(date_range, tuple) and len(date_range)==2
                      else (date_range, date_range))

    df_d = df[(df['tanggal'].astype('datetime64[ns]') >= pd.Timestamp(d_start)) &
              (df['tanggal'].astype('datetime64[ns]') <= pd.Timestamp(d_end))].copy()
    if filter_risk:
        df_d = df_d[df_d['risk_level'].isin(filter_risk)]
    if approver_f == 'TOLAK' and 'is_tolak' in df_d.columns:
        df_d = df_d[df_d['is_tolak']==1]
    elif approver_f == 'False Negative' and 'false_negative' in df_d.columns:
        df_d = df_d[df_d['false_negative']==1]
    elif approver_f == 'PENDING' and 'is_pending' in df_d.columns:
        df_d = df_d[df_d['is_pending']==1]

    if df_d.empty:
        st.warning("Tidak ada data pada rentang ini."); return

    n_fn_d = df_d['false_negative'].sum() if 'false_negative' in df_d.columns else 0
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card info"><div class="metric-val">{len(df_d):,}</div><div class="metric-lbl">Absensi</div></div>
        <div class="metric-card info"><div class="metric-val">{df_d['karyawan_id'].nunique()}</div><div class="metric-lbl">Karyawan</div></div>
        <div class="metric-card info"><div class="metric-val">{df_d['id_skpd'].nunique()}</div><div class="metric-lbl">SKPD</div></div>
        <div class="metric-card danger"><div class="metric-val">{(df_d['risk_level']=='HIGH').sum():,}</div><div class="metric-lbl">HIGH</div></div>
        <div class="metric-card {'danger' if n_fn_d>0 else ''}"><div class="metric-val">{n_fn_d}</div><div class="metric-lbl">False Neg.</div></div>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["🗺️ Peta Snapshot","⏰ Distribusi Jam","🚨 Daftar Karyawan","🔍 Deteksi Titipan"])

    with t1:
        MAX = 1500
        df_disp = df_d.sample(MAX, random_state=42) if len(df_d)>MAX else df_d
        if len(df_d)>MAX: st.info(f"Menampilkan {MAX:,} dari {len(df_d):,} titik.")
        m = folium.Map(location=[df_disp['lat'].median(), df_disp['long'].median()],
                       zoom_start=13, tiles='CartoDB positron')
        mc = MarkerCluster(name='Absensi').add_to(m)
        for _,row in df_disp.iterrows():
            c = risk_color_folium(row.get('risk_level','LOW'))
            ec = 'gold' if row.get('false_negative',0)==1 else c
            popup = f"""<div style='font-size:12px'>
                <b>ID {row.get('karyawan_id','')} | SKPD {row.get('id_skpd','')}</b><br>
                {str(row.get('tanggal_kirim',''))[:16]}<br>
                Jarak: {row.get('dist_km',0):.3f} km | Score: {row.get('anomaly_score',0)}<br>
                Risk: <b>{row.get('risk_level','')}</b> | Approver: {row.get('approver_status','-') or '-'}<br>
                {'⚠️ FALSE NEGATIVE' if row.get('false_negative',0)==1 else ''}
            </div>"""
            folium.CircleMarker(
                location=[row['lat'],row['long']],
                radius=9 if row.get('risk_level')=='HIGH' else 6,
                color=ec, fill=True, fill_color=c, fill_opacity=0.75,
                popup=folium.Popup(popup, max_width=230)
            ).add_to(mc)
        if not oc.empty:
            fg_off = folium.FeatureGroup(name='Kantor')
            for _,o in oc.iterrows():
                if pd.notna(o.get('office_lat')):
                    folium.Marker([o['office_lat'],o['office_long']],
                                  popup=f"SKPD {o['id_skpd']}",
                                  icon=folium.Icon(color='blue',icon='home',prefix='fa')).add_to(fg_off)
                    folium.Circle([o['office_lat'],o['office_long']], radius=300,
                                  color='blue', fill=False, weight=1.5, dash_array='5').add_to(fg_off)
            fg_off.add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=500, returned_objects=[])

    with t2:
        if 'jam' in df_d.columns:
            col_a,col_b = st.columns(2)
            with col_a:
                fig = px.bar(df_d.groupby(['jam','risk_level']).size().reset_index(name='n'),
                             x='jam', y='n', color='risk_level', barmode='stack',
                             title='Per Jam',
                             color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
                fig.add_vrect(x0=7,x1=9, fillcolor='green', opacity=0.07, annotation_text='Masuk')
                fig.add_vrect(x0=15,x1=17, fillcolor='purple', opacity=0.07, annotation_text='Pulang')
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                if len(df_d['tanggal'].unique()) > 1:
                    daily = df_d.groupby(['tanggal','risk_level']).size().reset_index(name='n')
                    fig2 = px.line(daily, x='tanggal', y='n', color='risk_level', markers=True,
                                   title='Trend Harian',
                                   color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
                    fig2.update_layout(height=360)
                    st.plotly_chart(fig2, use_container_width=True)

    with t3:
        emp_day = df_d.groupby(['karyawan_id','id_skpd']).agg(
            n_absensi=('karyawan_id','count'),
            n_high=('risk_level', lambda x: (x=='HIGH').sum()),
            max_score=('anomaly_score','max'),
            avg_dist=('dist_km','mean'),
            n_tolak=('is_tolak','sum') if 'is_tolak' in df_d.columns else ('karyawan_id', lambda x: 0),
            n_fn=('false_negative','sum') if 'false_negative' in df_d.columns else ('karyawan_id', lambda x: 0),
        ).reset_index().sort_values('n_high', ascending=False)

        col_f1,col_f2 = st.columns(2)
        with col_f1:
            min_high = st.slider("Min HIGH risk", 0, max(1,int(emp_day['n_high'].max())), 1, key='hunt_dt_mh')
        with col_f2:
            sel_skpd_dt = st.selectbox("Filter SKPD", ['Semua']+sorted(df_d['id_skpd'].unique().tolist()), key='hunt_dt_sk')

        emp_f = emp_day[emp_day['n_high'] >= min_high]
        if sel_skpd_dt != 'Semua':
            emp_f = emp_f[emp_f['id_skpd']==sel_skpd_dt]

        st.caption(f"{len(emp_f)} karyawan")
        st.dataframe(emp_f, use_container_width=True, height=380)
        csv = emp_f.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download", csv, f"karyawan_{d_start}_{d_end}.csv","text/csv")

    with t4:
        st.markdown("#### 🔍 Deteksi Absensi Titipan")
        st.caption("Cari karyawan berbeda yang absen di lokasi sangat berdekatan pada waktu hampir sama.")
        col_r1,col_r2 = st.columns(2)
        with col_r1: radius_m  = st.slider("Radius lokasi (meter)", 5, 200, 50, 5, key='hunt_dt_rad')
        with col_r2: window_m  = st.slider("Jendela waktu (menit)", 1, 60, 15, 1, key='hunt_dt_win')

        n_check = len(df_d)
        est_pairs = n_check*(n_check-1)//2
        if est_pairs > 500000:
            st.warning(f"⚠️ Data terlalu banyak ({n_check:,} baris, ~{est_pairs:,} pasangan). Perkecil rentang tanggal.")
        elif st.button("🔍 Deteksi Sekarang", type="primary", key='hunt_dt_det'):
            with st.spinner("⏳ Mendeteksi..."):
                df_ck = df_d[['karyawan_id','id_skpd','lat','long','tanggal_kirim','jenis',
                               'anomaly_score','risk_level']].dropna().copy()
                rows_l = df_ck.to_dict('records')
                n = len(rows_l)
                suspicious = []
                for i in range(n):
                    for j in range(i+1, n):
                        r1,r2 = rows_l[i],rows_l[j]
                        if r1['karyawan_id']==r2['karyawan_id']: continue
                        dm = haversine_scalar(r1['lat'],r1['long'],r2['lat'],r2['long'])
                        if dm > radius_m: continue
                        dt = abs((pd.Timestamp(r1['tanggal_kirim'])-pd.Timestamp(r2['tanggal_kirim'])).total_seconds())/60
                        if dt <= window_m:
                            suspicious.append({
                                'Karyawan A': r1['karyawan_id'], 'Karyawan B': r2['karyawan_id'],
                                'SKPD A': r1['id_skpd'], 'SKPD B': r2['id_skpd'],
                                'Jarak (m)': round(dm,1), 'Selisih Waktu (mnt)': round(dt,1),
                                'Lat': round(r1['lat'],5), 'Long': round(r1['long'],5),
                                'Waktu A': str(r1['tanggal_kirim'])[:16],
                                'Waktu B': str(r2['tanggal_kirim'])[:16],
                                'Score A': r1['anomaly_score'], 'Score B': r2['anomaly_score'],
                            })
                        if len(suspicious) >= 300: break
                    if len(suspicious) >= 300: break
                st.session_state['kolusi_result'] = suspicious

        if 'kolusi_result' in st.session_state:
            kol = st.session_state['kolusi_result']
            if not kol:
                st.success("✅ Tidak ditemukan indikasi titipan.")
            else:
                st.error(f"🚨 {len(kol)} pasangan absensi mencurigakan!")
                df_kol = pd.DataFrame(kol).sort_values('Jarak (m)')
                # Peta titipan
                m_k = folium.Map(location=[df_d['lat'].median(), df_d['long'].median()],
                                 zoom_start=14, tiles='CartoDB positron')
                for _,kr in df_kol.head(50).iterrows():
                    folium.CircleMarker(
                        location=[kr['Lat'],kr['Long']], radius=13,
                        color='darkred', fill=True, fill_color='#e74c3c', fill_opacity=0.7,
                        popup=f"{kr['Karyawan A']} & {kr['Karyawan B']}<br>{kr['Jarak (m)']}m | {kr['Selisih Waktu (mnt)']} mnt"
                    ).add_to(m_k)
                st_folium(m_k, width=None, height=360, returned_objects=[])
                st.dataframe(df_kol, use_container_width=True, height=320)
                csv_k = df_kol.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Hasil Deteksi", csv_k, "titipan.csv","text/csv")

# ============================================================
# PAGE: PREDIKSI
# ============================================================
def page_prediksi():
    st.markdown("## 🔮 Prediksi Absensi Baru")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Upload data dulu di menu **📥 Upload Data**."); return

    oc = st.session_state.get('office_centroid', pd.DataFrame())
    if oc.empty:
        st.error("Data kantor SKPD tidak tersedia."); return

    st.info(f"✅ Referensi kantor tersedia untuk **{len(oc)} SKPD**")

    pred_file = st.file_uploader("Upload file baru (opsional)", type=['csv','xlsx'], key='pred_up')

    st.markdown("### ✏️ Input Manual")
    with st.form("manual_pred"):
        c1,c2,c3 = st.columns(3)
        with c1:
            m_kar  = st.number_input("Karyawan ID", min_value=1, value=1001)
            m_skpd = st.selectbox("ID SKPD", sorted(oc['id_skpd'].tolist()))
        with c2:
            m_lat  = st.number_input("Latitude",  value=float(oc['office_lat'].median()), format="%.6f")
            m_long = st.number_input("Longitude", value=float(oc['office_long'].median()), format="%.6f")
        with c3:
            m_jenis   = st.selectbox("Jenis", ['M','P'], format_func=lambda x:'Masuk' if x=='M' else 'Pulang')
            m_waktu   = st.time_input("Jam Absensi")
            m_catatan = st.text_input("Catatan (kosong jika tidak ada)", "")
            m_approver = st.selectbox("Approver Status", ['','TERIMA','TOLAK'])
        submitted = st.form_submit_button("🔮 Prediksi", type="primary", use_container_width=True)

    if submitted:
        df_p = pd.DataFrame([{
            'karyawan_id': m_kar, 'id_skpd': m_skpd,
            'lat': m_lat, 'long': m_long, 'jenis': m_jenis,
            'tanggal_kirim': pd.Timestamp.now().replace(hour=m_waktu.hour, minute=m_waktu.minute, second=0),
            'catatan': m_catatan or np.nan, 'status_lokasi': 1,
            'approver_status': m_approver
        }])
        _run_prediction(df_p, oc, batch=False)

    if pred_file:
        fb = pred_file.getvalue()
        df_pf, _ = load_processed_file(fb, pred_file.name)
        st.markdown(f"### Prediksi Batch — {len(df_pf):,} baris")
        if st.button("🚀 Jalankan Prediksi Batch", type="primary"):
            with st.spinner("⏳ Memproses..."):
                _run_prediction(df_pf, oc, batch=True)

def _run_prediction(df_in, oc, batch=False):
    df = df_in.copy()
    if 'tanggal_kirim' in df.columns:
        df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'], errors='coerce')
    if 'catatan' not in df.columns: df['catatan'] = np.nan
    if 'status_lokasi' not in df.columns: df['status_lokasi'] = 1

    df = df.merge(oc, on='id_skpd', how='left')
    df['dist_km'] = df.apply(
        lambda r: haversine_scalar(r['lat'],r['long'],
                                   r.get('office_lat',r['lat']),
                                   r.get('office_long',r['long']))/1000
        if pd.notna(r.get('office_lat')) else 0.0, axis=1)

    df['outside_300m']     = (df['dist_km']>0.3).astype(int)
    df['very_far']         = (df['dist_km']>5.0).astype(int)
    df['extreme_far']      = (df['dist_km']>50.0).astype(int)
    df['no_note']          = df['catatan'].isna().astype(int)
    df['far_no_note']      = ((df['outside_300m']==1)&(df['no_note']==1)).astype(int)
    df['near_but_status0'] = ((df['outside_300m']==0)&(df['status_lokasi']==0)).astype(int)
    df['anomaly_score']    = (df['outside_300m']*25 + df['far_no_note']*35 +
                              df['very_far']*30 + df['extreme_far']*50 +
                              df['near_but_status0']*5)
    df['risk_level']       = df['anomaly_score'].apply(lambda s: 'HIGH' if s>=70 else 'MED' if s>=30 else 'LOW')
    df['system_action']    = df['risk_level'].apply(
        lambda r: '✅ AUTO APPROVE' if r=='LOW' else '⏸️ HOLD' if r=='MED' else '❌ TEMP REJECT')

    # Approver flags
    if 'approver_status' in df.columns:
        df['is_tolak']       = df['approver_status'].astype(str).str.upper().str.contains('TOLAK', na=False).astype(int)
        df['is_terima']      = df['approver_status'].astype(str).str.upper().str.contains('TERIMA', na=False).astype(int)
        df['false_negative'] = ((df['is_terima']==1)&(df['risk_level']=='HIGH')).astype(int)

    if not batch:
        row = df.iloc[0]
        risk  = row['risk_level']
        color = risk_color_hex(risk)
        st.markdown("---"); st.markdown("### 🎯 Hasil Prediksi")
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Jarak ke Kantor", f"{row['dist_km']:.3f} km")
        with c2: st.metric("Anomaly Score",   int(row['anomaly_score']))
        with c3: st.metric("Risk Level",       risk)
        action = row['system_action']
        (st.success if risk=='LOW' else st.warning if risk=='MED' else st.error)(f"**Aksi Sistem:** {action}")
        fn = row.get('false_negative',0)
        if fn:
            st.error("🚨 FALSE NEGATIVE — TERIMA tapi HIGH risk! Perlu review manual.")
        sigs = {'Di luar 300m':int(row['outside_300m']),'Jauh tanpa catatan':int(row['far_no_note']),
                'Sangat jauh >5km':int(row['very_far']),'Extreme >50km':int(row['extreme_far'])}
        sig_df = pd.DataFrame({'Sinyal':list(sigs.keys()),
                               'Status':['⚠️ YA' if v else '✅ TIDAK' for v in sigs.values()]})
        st.dataframe(sig_df, use_container_width=True, hide_index=True)
    else:
        st.success(f"✅ Selesai untuk {len(df):,} baris")
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("HIGH", f"{(df['risk_level']=='HIGH').sum():,}")
        with c2: st.metric("MED",  f"{(df['risk_level']=='MED').sum():,}")
        with c3: st.metric("LOW",  f"{(df['risk_level']=='LOW').sum():,}")
        with c4:
            fn_b = df.get('false_negative', pd.Series([0]*len(df))).sum() if 'false_negative' in df.columns else 0
            st.metric("False Neg.", f"{fn_b:,}")
        cols_show = ['karyawan_id','id_skpd','jenis','tanggal_kirim','lat','long',
                     'dist_km','anomaly_score','risk_level','approver_status','false_negative','system_action']
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(df[cols_show].sort_values('anomaly_score',ascending=False), use_container_width=True)
        csv = df[cols_show].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil", csv, "prediksi.csv","text/csv")

# ============================================================
# MAIN
# ============================================================
def main():
    if 'df' not in st.session_state:             st.session_state['df'] = None
    if 'office_centroid' not in st.session_state: st.session_state['office_centroid'] = pd.DataFrame()
    if 'watchlist' not in st.session_state:       st.session_state['watchlist'] = []

    # ── AUTO-LOAD FILE LOKAL (saat pertama buka app) ──────────
    if st.session_state['df'] is None and not st.session_state.get('_autoload_attempted'):
        st.session_state['_autoload_attempted'] = True
        local_files = scan_local_files()
        if local_files:
            chosen = local_files[0]
            try:
                df, fixed = load_local_file(chosen)
                st.session_state['df']              = df
                st.session_state['office_centroid'] = build_office_centroid(df)
                st.session_state['_file_hash']      = chosen
                st.session_state['file_name']       = chosen
                st.session_state['_autoloaded']     = chosen
            except Exception:
                pass

    page, uploaded, filters = render_sidebar()

    # ── AUTO-LOAD FILE UPLOAD BARU ────────────────────────────
    if uploaded is not None:
        fhash = hashlib.md5(uploaded.getvalue()).hexdigest()
        if st.session_state.get('_file_hash') != fhash:
            file_bytes = uploaded.getvalue()
            df, fixed = load_processed_file(file_bytes, uploaded.name)
            st.session_state['df']              = df
            st.session_state['office_centroid'] = build_office_centroid(df)
            st.session_state['_file_hash']      = fhash
            st.session_state['file_name']       = uploaded.name
            st.session_state.pop('_autoloaded', None)

    # ── BANNER FILE AKTIF DI SIDEBAR ─────────────────────────
    if st.session_state.get('_autoloaded') and st.session_state['df'] is not None:
        fname = st.session_state['_autoloaded']
        n     = len(st.session_state['df'])
        st.sidebar.success(f"\u2705 Auto-loaded: **{fname}**\n{n:,} baris")

    if page == "🏠 Beranda":
        page_beranda()
    elif page == "📥 Upload Data":
        page_upload(uploaded)
    elif page == "📊 Visualisasi":
        page_visualisasi(filters)
    elif page == "🎯 Hunting":
        page_hunting()
    elif page == "🔮 Prediksi":
        page_prediksi()

    st.markdown("---")
    st.markdown('<p style="text-align:center;color:gray;font-size:11px">'
                'Deteksi Anomali Absensi | DBSCAN + ST-DBSCAN + Haversine | Streamlit + Folium + Plotly'
                '</p>', unsafe_allow_html=True)
if __name__ == '__main__':
    main()
