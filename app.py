"""
Visualisasi & Analisis Absensi v3
Status Presensi: T2/T3/T4/TWM/TWP/PC1-4 dari kolom status_presensi
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
import io, hashlib, os
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

st.set_page_config(
    page_title="Analisis Absensi",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# KONSTANTA STATUS
# ============================================================

# Mapping KODE MENTAH dari kolom status_presensi → label baru
# Hanya kode-kode yang memang muncul di data (T2, T3, T4, TWM, TWP, PC1-PC4)
STATUS_CODE_MAP = {
    'T2':  'TELAT_RINGAN',
    'T3':  'TELAT_SEDANG',
    'T4':  'TELAT_BERAT',
    'TWM': 'TEPAT_WAKTU_MASUK',
    'TWP': 'TEPAT_WAKTU_PULANG',
    'PC1': 'PULANG_CEPAT',
    'PC2': 'PULANG_CEPAT_RINGAN',
    'PC3': 'PULANG_CEPAT_SEDANG',
    'PC4': 'PULANG_CEPAT_BERAT',
}

# Label lama dari preprocessing sebelumnya → label baru
STATUS_LEGACY_MAP = {
    'PULANG_NORMAL': 'TEPAT_WAKTU_PULANG',
    'HADIR':         'TEPAT_WAKTU_MASUK',
}

# Label yang sudah benar — tidak perlu diubah
STATUS_VALID = {
    'TELAT_RINGAN', 'TELAT_SEDANG', 'TELAT_BERAT',
    'TEPAT_WAKTU_MASUK', 'TEPAT_WAKTU_PULANG',
    'PULANG_CEPAT', 'PULANG_CEPAT_RINGAN', 'PULANG_CEPAT_SEDANG', 'PULANG_CEPAT_BERAT',
}

# Label ambigu yang butuh konteks tambahan untuk disambiguasi
STATUS_AMBIGUOUS = {'TELAT', 'PULANG'}

STATUS_ORDER = [
    'TELAT_BERAT', 'PULANG_CEPAT_BERAT',
    'TELAT_SEDANG', 'PULANG_CEPAT_SEDANG',
    'TELAT_RINGAN', 'PULANG_CEPAT_RINGAN', 'PULANG_CEPAT',
    'TEPAT_WAKTU_MASUK', 'TEPAT_WAKTU_PULANG',
]

STATUS_BERMASALAH = {
    'TELAT_BERAT', 'TELAT_SEDANG',
    'PULANG_CEPAT_BERAT', 'PULANG_CEPAT_SEDANG',
}

STATUS_COLORS = {
    'TELAT_BERAT':         '#c0392b',
    'TELAT_SEDANG':        '#e67e22',
    'TELAT_RINGAN':        '#d4ac0d',
    'TEPAT_WAKTU_MASUK':   '#27ae60',
    'TEPAT_WAKTU_PULANG':  '#2ecc71',
    'PULANG_CEPAT':        '#f39c12',
    'PULANG_CEPAT_RINGAN': '#d4ac0d',
    'PULANG_CEPAT_SEDANG': '#e67e22',
    'PULANG_CEPAT_BERAT':  '#c0392b',
    'UNKNOWN':             '#95a5a6',
}

STATUS_EMOJI = {
    'TELAT_BERAT':         '🔴',
    'TELAT_SEDANG':        '🟠',
    'TELAT_RINGAN':        '🟡',
    'TEPAT_WAKTU_MASUK':   '🟢',
    'TEPAT_WAKTU_PULANG':  '🟢',
    'PULANG_CEPAT':        '🟡',
    'PULANG_CEPAT_RINGAN': '🟡',
    'PULANG_CEPAT_SEDANG': '🟠',
    'PULANG_CEPAT_BERAT':  '🔴',
    'UNKNOWN':             '⚪',
}

STATUS_FOLIUM = {
    'TELAT_BERAT':         'red',
    'TELAT_SEDANG':        'orange',
    'TELAT_RINGAN':        'lightred',
    'TEPAT_WAKTU_MASUK':   'green',
    'TEPAT_WAKTU_PULANG':  'green',
    'PULANG_CEPAT':        'lightred',
    'PULANG_CEPAT_RINGAN': 'lightred',
    'PULANG_CEPAT_SEDANG': 'orange',
    'PULANG_CEPAT_BERAT':  'red',
    'UNKNOWN':             'gray',
}

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:bold;color:#1f77b4;text-align:center;padding:1rem 0 .2rem}
.sub-header{text-align:center;color:#666;font-size:.95rem;margin-bottom:1.5rem}
.hunt-header{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:1.4rem 2rem;
  border-radius:12px;margin-bottom:1.5rem;border:1px solid #e94560}
.hunt-title{font-size:1.7rem;font-weight:900;color:#e94560;font-family:'Courier New',monospace;
  letter-spacing:2px;margin:0}
.hunt-sub{color:#a8b2d8;font-size:.83rem;margin-top:.3rem;font-family:'Courier New',monospace}
.section-header{display:flex;align-items:center;gap:.8rem;background:#f8f9fa;
  border-left:5px solid #1f77b4;padding:.8rem 1.2rem;border-radius:0 10px 10px 0;margin:1.2rem 0 1rem}
.metric-grid{display:flex;gap:.8rem;flex-wrap:wrap;margin:.8rem 0}
.metric-card{flex:1;min-width:110px;background:white;border:1px solid #e0e0e0;
  border-radius:10px;padding:.9rem;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.06)}
.metric-val{font-size:1.45rem;font-weight:800;color:#2c3e50}
.metric-lbl{font-size:.68rem;color:#7f8c8d;margin-top:2px;text-transform:uppercase;letter-spacing:.4px}
.mc-red .metric-val{color:#c0392b}
.mc-orange .metric-val{color:#e67e22}
.mc-green .metric-val{color:#27ae60}
.mc-blue .metric-val{color:#3498db}
.alert-box{border-radius:8px;padding:.9rem 1.1rem;margin:.6rem 0;font-size:.88rem}
.alert-red{background:#fff5f5;border:1px solid #fc8181;color:#c53030}
.alert-blue{background:#ebf8ff;border:1px solid #90cdf4;color:#2b6cb0}
.alert-orange{background:#fffaf0;border:1px solid #f6ad55;color:#c05621}
.alert-green{background:#f0fff4;border:1px solid #9ae6b4;color:#276749}
.watchlist-item{display:flex;align-items:center;gap:.7rem;background:white;
  border:1px solid #fde8e8;border-left:4px solid #c0392b;border-radius:7px;
  padding:.6rem .9rem;margin-bottom:.4rem;font-size:.85rem}
.remap-badge{display:inline-block;background:#ebf8ff;border:1px solid #90cdf4;
  color:#2b6cb0;border-radius:6px;padding:2px 8px;font-size:.78rem;margin:2px}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CORE MAPPING FUNCTION
# ============================================================

def map_status_value(val):
    """
    Konversi 1 nilai dari kolom status_presensi ke label standar.

    Urutan prioritas:
    1. Kode mentah mesin (T2, T3, T4, TWM, TWP, PC1-PC4)
    2. Label valid yang sudah benar
    3. Label lama / legacy (PULANG_NORMAL, HADIR)
    4. Label ambigu (TELAT, PULANG) → dikembalikan apa adanya, dihandle terpisah
    5. Selain itu → UNKNOWN
    """
    if pd.isna(val) or str(val).strip() == '':
        return 'UNKNOWN'
    v = str(val).strip().upper()
    # 1. Kode mentah
    if v in STATUS_CODE_MAP:
        return STATUS_CODE_MAP[v]
    # 2. Label valid
    if v in STATUS_VALID:
        return v
    # 3. Legacy
    if v in STATUS_LEGACY_MAP:
        return STATUS_LEGACY_MAP[v]
    # 4. Ambigu — kembalikan apa adanya, disambiguasi di resolve_ambiguous()
    if v in STATUS_AMBIGUOUS:
        return v
    return 'UNKNOWN'


def resolve_ambiguous(df):
    """
    Disambiguasi label TELAT dan PULANG yang masih tersisa setelah map_status_value().
    - TELAT  → TEPAT_WAKTU_MASUK / TEPAT_WAKTU_PULANG  (berdasarkan kolom jenis)
    - PULANG → PULANG_CEPAT_SEDANG / PULANG_CEPAT_BERAT (berdasarkan kolom jam_desimal)
    """
    remaps = []

    # TELAT → TEPAT_WAKTU
    mask = df['status_presensi'] == 'TELAT'
    if mask.any():
        n = mask.sum()
        if 'jenis' in df.columns:
            df.loc[mask & (df['jenis'] == 'M'), 'status_presensi'] = 'TEPAT_WAKTU_MASUK'
            df.loc[mask & (df['jenis'] == 'P'), 'status_presensi'] = 'TEPAT_WAKTU_PULANG'
            # Sisa yang jenis-nya tidak M/P → default masuk
            df.loc[df['status_presensi'] == 'TELAT', 'status_presensi'] = 'TEPAT_WAKTU_MASUK'
        else:
            df.loc[mask, 'status_presensi'] = 'TEPAT_WAKTU_MASUK'
        remaps.append(f"🔄 {n:,} baris 'TELAT' → TEPAT_WAKTU (berdasarkan kolom jenis)")

    # PULANG → PULANG_CEPAT
    mask = df['status_presensi'] == 'PULANG'
    if mask.any():
        n = mask.sum()
        if 'jam_desimal' in df.columns:
            df.loc[mask & (df['jam_desimal'] >= 14), 'status_presensi'] = 'PULANG_CEPAT_SEDANG'
            df.loc[mask & (df['jam_desimal'] < 14),  'status_presensi'] = 'PULANG_CEPAT_BERAT'
            df.loc[df['status_presensi'] == 'PULANG', 'status_presensi'] = 'PULANG_CEPAT_SEDANG'
        else:
            df.loc[mask, 'status_presensi'] = 'PULANG_CEPAT_SEDANG'
        remaps.append(f"🔄 {n:,} baris 'PULANG' → PULANG_CEPAT (berdasarkan jam_desimal)")

    return df, remaps


def determine_status_from_jam(jam_desimal, jenis):
    """Derive status dari jam & jenis jika kolom status_presensi tidak ada."""
    if jenis == 'M':
        if jam_desimal <= 7.5:   return 'TEPAT_WAKTU_MASUK'
        elif jam_desimal <= 8.0: return 'TELAT_RINGAN'
        elif jam_desimal <= 9.0: return 'TELAT_SEDANG'
        else:                    return 'TELAT_BERAT'
    else:
        if jam_desimal >= 16.0:   return 'TEPAT_WAKTU_PULANG'
        elif jam_desimal >= 15.5: return 'PULANG_CEPAT'
        elif jam_desimal >= 15.0: return 'PULANG_CEPAT_RINGAN'
        elif jam_desimal >= 14.0: return 'PULANG_CEPAT_SEDANG'
        else:                     return 'PULANG_CEPAT_BERAT'

# ============================================================
# HELPERS
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def status_color(s):   return STATUS_COLORS.get(s, '#95a5a6')
def status_emoji(s):   return STATUS_EMOJI.get(s, '⚪')
def status_folium_color(s): return STATUS_FOLIUM.get(s, 'gray')
def is_bermasalah(s):  return s in STATUS_BERMASALAH

def _df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# ============================================================
# FIX DECIMAL
# ============================================================

def fix_decimal_columns(df):
    numeric_hints = [
        'lat', 'long', 'lat_rad', 'long_rad', 'office_lat', 'office_long',
        'dist_km', 'jarak', 'jam_desimal', 'jam', 'menit', 'weekday',
        'outside_300m', 'very_far', 'extreme_far', 'status_lokasi', 'timestamp_num',
    ]
    fixed = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if col in numeric_hints:
            try:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.', regex=False).str.strip(),
                    errors='coerce')
                fixed.append(col)
            except Exception:
                pass
        else:
            sample = df[col].dropna().head(20).astype(str)
            if sample.str.match(r'^-?\d+,\d+$').mean() > 0.7:
                try:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '.', regex=False),
                        errors='coerce')
                    fixed.append(col)
                except Exception:
                    pass
    return df, fixed

# ============================================================
# LOAD & PROCESS
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_processed_file(file_bytes, file_name):
    buf = io.BytesIO(file_bytes)
    if file_name.endswith('.csv'):
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

    df, fixed_cols = fix_decimal_columns(df)

    # Datetime
    if 'tanggal_kirim' in df.columns:
        df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'], errors='coerce')
        if 'jam' not in df.columns:
            df['jam'] = df['tanggal_kirim'].dt.hour
        if 'menit' not in df.columns:
            df['menit'] = df['tanggal_kirim'].dt.minute
        if 'jam_desimal' not in df.columns:
            df['jam_desimal'] = df['jam'] + df['menit'] / 60.0
        if 'weekday' not in df.columns:
            df['weekday'] = df['tanggal_kirim'].dt.weekday
        if 'tanggal' not in df.columns:
            df['tanggal'] = df['tanggal_kirim'].dt.date

    if 'jenis' in df.columns:
        df['jenis'] = df['jenis'].astype(str).str.strip().str.upper()

    # ════════════════════════════════════════════════════════
    # STATUS MAPPING — 3 TAHAP
    # ════════════════════════════════════════════════════════
    remaps = []

    if 'status_presensi' in df.columns:
        # TAHAP 1: Map kode mentah + label valid + legacy
        original_unique = df['status_presensi'].dropna().unique().tolist()
        df['status_presensi'] = df['status_presensi'].apply(map_status_value)

        # Catat apa yang dimapping
        for orig in original_unique:
            mapped = map_status_value(orig)
            if str(orig).strip().upper() != mapped and mapped not in STATUS_AMBIGUOUS:
                remaps.append(f"<span class='remap-badge'>{orig} → {mapped}</span>")

        # TAHAP 2: Disambiguasi TELAT / PULANG
        df, ambig_remaps = resolve_ambiguous(df)
        remaps.extend(ambig_remaps)

        # TAHAP 3: UNKNOWN yang bisa di-derive dari jam
        mask_unk = df['status_presensi'].isin(['UNKNOWN'])
        if mask_unk.any() and 'jam_desimal' in df.columns and 'jenis' in df.columns:
            n_derived = mask_unk.sum()
            df.loc[mask_unk, 'status_presensi'] = df.loc[mask_unk].apply(
                lambda r: determine_status_from_jam(r['jam_desimal'], r['jenis']), axis=1)
            remaps.append(f"🔄 {n_derived:,} baris UNKNOWN → derive dari jam_desimal")

    elif 'jam_desimal' in df.columns and 'jenis' in df.columns:
        df['status_presensi'] = df.apply(
            lambda r: determine_status_from_jam(r['jam_desimal'], r['jenis']), axis=1)
        remaps.append("🔄 Kolom status_presensi tidak ada → derive semua dari jam_desimal")
    else:
        df['status_presensi'] = 'UNKNOWN'

    # Flags
    df['is_bermasalah'] = df['status_presensi'].apply(
        lambda s: 1 if s in STATUS_BERMASALAH else 0)

    if 'approver_status' in df.columns:
        df['approver_status'] = (df['approver_status'].astype(str).str.strip()
                                 .replace({'nan': '', 'None': '', 'NaN': ''}))
        df['is_tolak']   = df['approver_status'].str.upper().str.contains('TOLAK', na=False).astype(int)
        df['is_terima']  = df['approver_status'].str.upper().str.contains('TERIMA', na=False).astype(int)
        df['is_pending'] = ((df['approver_status'] == '') | df['approver_status'].isna()).astype(int)
        df['terima_bermasalah'] = ((df['is_terima'] == 1) & (df['is_bermasalah'] == 1)).astype(int)
    else:
        df['is_tolak'] = df['is_terima'] = df['is_pending'] = df['terima_bermasalah'] = 0

    if 'dist_km' in df.columns:
        df['outside_300m'] = (df['dist_km'] > 0.3).astype(int)
        df['very_far']     = (df['dist_km'] > 5.0).astype(int)

    return df, fixed_cols, remaps


def build_office_centroid(df):
    if 'office_lat' in df.columns and 'id_skpd' in df.columns:
        return df.groupby('id_skpd')[['office_lat', 'office_long']].first().reset_index()
    return pd.DataFrame(columns=['id_skpd', 'office_lat', 'office_long'])


def validate_dataframe(df):
    warns = []
    required = ['karyawan_id', 'lat', 'long', 'tanggal_kirim', 'jenis', 'id_skpd']
    missing = [c for c in required if c not in df.columns]
    if missing:
        warns.append(f"Kolom wajib tidak ditemukan: {missing}")
    if 'lat' in df.columns and (~df['lat'].between(-90, 90)).sum() > 0:
        warns.append(f"{(~df['lat'].between(-90, 90)).sum()} baris lat di luar range")
    return warns

# ============================================================
# FILTER
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def apply_filters(df_hash, df, skpd, jenis_tuple, date_range,
                  dist_range, approver_filter, status_filter):
    f = df.copy()
    if skpd != 'Semua':
        f = f[f['id_skpd'] == skpd]
    if jenis_tuple:
        f = f[f['jenis'].isin(list(jenis_tuple))]
    if date_range and len(date_range) == 2 and 'tanggal_kirim' in f.columns:
        f = f[(f['tanggal_kirim'].dt.date >= date_range[0]) &
              (f['tanggal_kirim'].dt.date <= date_range[1])]
    if 'dist_km' in f.columns:
        f = f[(f['dist_km'] >= dist_range[0]) & (f['dist_km'] <= dist_range[1])]
    if approver_filter and approver_filter != 'Semua':
        if approver_filter == 'TOLAK':       f = f[f['is_tolak'] == 1]
        elif approver_filter == 'TERIMA':    f = f[f['is_terima'] == 1]
        elif approver_filter == 'PENDING':   f = f[f['is_pending'] == 1]
        elif approver_filter == 'Terima Bermasalah': f = f[f['terima_bermasalah'] == 1]
    if status_filter:
        f = f[f['status_presensi'].isin(status_filter)]
    return f

# ============================================================
# LOCAL FILES
# ============================================================

CANDIDATE_FILES = [
    'dataset_absensi_final2.xlsx', 'absen_pegawai.xlsx',
    'absensi.xlsx', 'absensi_processed.xlsx', 'absensi_processed.csv',
    'absensi.csv', 'data_absensi.xlsx', 'data_absensi.csv',
]

def scan_local_files():
    found = []
    for fn in CANDIDATE_FILES:
        if os.path.exists(fn): found.append(fn)
    for fn in sorted(os.listdir('.')):
        if fn.endswith(('.xlsx', '.csv')) and fn not in found: found.append(fn)
    return found

@st.cache_data(show_spinner=False, ttl=3600)
def load_local_file(filepath):
    with open(filepath, 'rb') as f:
        fb = f.read()
    return load_processed_file(fb, filepath.split('/')[-1].split('\\')[-1])

# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    st.sidebar.markdown("## 🗺️ Analisis Absensi")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("📌 Navigasi", [
        "🏠 Beranda", "📥 Upload Data", "🔧 Preprocessing", "📊 Visualisasi", "🎯 Hunting", "🔮 Prediksi"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Upload Data")
    uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=['csv', 'xlsx'])
    filters = {}
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        with st.sidebar.expander("🔍 Filter Data", expanded=True):
            skpd_list = ['Semua'] + sorted(df['id_skpd'].unique().tolist())
            filters['skpd'] = st.selectbox("SKPD", skpd_list)
            all_status = sorted(df['status_presensi'].dropna().unique().tolist())
            filters['status'] = st.multiselect(
                "Status Presensi", all_status, default=all_status,
                format_func=lambda x: f"{status_emoji(x)} {x}")
            filters['jenis'] = st.multiselect(
                "Jenis", ['M', 'P'], default=['M', 'P'],
                format_func=lambda x: 'Masuk' if x == 'M' else 'Pulang')
            if 'tanggal_kirim' in df.columns:
                mn = df['tanggal_kirim'].min().date()
                mx = df['tanggal_kirim'].max().date()
                filters['date'] = st.date_input("Rentang Tanggal", value=(mn, mx),
                                                min_value=mn, max_value=mx)
            else:
                filters['date'] = None
            mx_d = float(df['dist_km'].max()) if 'dist_km' in df.columns else 100.0
            filters['dist'] = st.slider("Jarak (km)", 0.0,
                                        min(mx_d, 100.0), (0.0, min(mx_d, 100.0)), 0.1)
            if 'approver_status' in df.columns:
                filters['approver'] = st.selectbox(
                    "Approver", ['Semua', 'TERIMA', 'TOLAK', 'PENDING', 'Terima Bermasalah'])
            else:
                filters['approver'] = 'Semua'
        with st.sidebar.expander("🗺️ Peta", expanded=False):
            filters['map_type'] = st.radio("Tipe", ['marker', 'cluster', 'heatmap'],
                                           format_func=lambda x: {
                                               'marker': '📍 Marker',
                                               'cluster': '🔵 Cluster',
                                               'heatmap': '🔥 Heatmap'}[x])
        wl = st.session_state.get('watchlist', [])
        if wl:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👁️ Watchlist")
            for eid in wl:
                ed = df[df['karyawan_id'] == eid]
                nb = ed['is_bermasalah'].sum() if not ed.empty else 0
                st.sidebar.markdown(f"""<div class='watchlist-item'>
                    <span>🔴</span><span><b>ID {eid}</b> — {nb} bermasalah</span>
                </div>""", unsafe_allow_html=True)
            if st.sidebar.button("🗑️ Clear Watchlist"):
                st.session_state['watchlist'] = []
                st.rerun()
    return page, uploaded, filters

# ============================================================
# BERANDA
# ============================================================

def page_beranda():
    st.markdown('<div class="main-header">🗺️ Analisis Absensi Pegawai</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload data → status otomatis terpetakan dari kode mesin</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.info("### 📥 Step 1\n**Upload Data**")
    with c2: st.success("### 📊 Step 2\n**Visualisasi**")
    with c3: st.warning("### 🎯 Step 3\n**Hunting**")
    with c4: st.error("### 🔮 Step 4\n**Prediksi**")

    st.markdown("---")
    st.markdown("### 📋 Mapping Kode → Label (dari kolom `status_presensi`)")

    st.markdown("**MASUK** — diukur dari `masuk_post_time` (batas toleransi akhir masuk):")
    masuk_df = pd.DataFrame([
        ['TWM', 'TEPAT_WAKTU_MASUK',   '🟢', '≤ 0 menit',   'Absen jam 08:10, toleransi s/d 08:15 → aman'],
        ['T2',  'TELAT_RINGAN',        '🟡', '0 – 14 menit', 'Absen jam 08:20 → telat 4 menit'],
        ['T3',  'TELAT_SEDANG',        '🟠', '14 – 44 menit','Absen jam 08:40 → telat 25 menit'],
        ['T4',  'TELAT_BERAT',         '🔴', '> 44 menit',   'Absen jam 10:00 → telat 104 menit'],
    ], columns=['Kode', 'Label', '', 'Durasi Telat', 'Contoh'])
    st.dataframe(masuk_df, use_container_width=True, hide_index=True)

    st.markdown("**PULANG** — diukur dari `pulang_pre_time` (jam pulang minimum):")
    pulang_df = pd.DataFrame([
        ['TWP', 'TEPAT_WAKTU_PULANG',  '🟢', '≤ 0 menit',    'Pulang jam 16:35, minimum 16:30 → aman'],
        ['PC1', 'PULANG_CEPAT',        '🟡', '0 – 30 menit',  'Pulang jam 16:20 → 10 menit terlalu cepat'],
        ['PC2', 'PULANG_CEPAT_RINGAN', '🟡', '30 – 60 menit', 'Pulang jam 15:45 → 45 menit terlalu cepat'],
        ['PC3', 'PULANG_CEPAT_SEDANG', '🟠', '60 – 90 menit', 'Pulang jam 15:10 → 80 menit terlalu cepat'],
        ['PC4', 'PULANG_CEPAT_BERAT',  '🔴', '> 90 menit',    'Pulang jam 13:00 → 210 menit terlalu cepat'],
    ], columns=['Kode', 'Label', '', 'Durasi Pulang Cepat', 'Contoh'])
    st.dataframe(pulang_df, use_container_width=True, hide_index=True)

    st.markdown("### 📋 Kolom yang Dibutuhkan")
    st.dataframe(pd.DataFrame([
        ['karyawan_id',    'integer',  'ID pegawai',                  'Wajib'],
        ['id_skpd',        'integer',  'ID kantor/SKPD',              'Wajib'],
        ['lat / long',     'float',    'Koordinat absensi',           'Wajib'],
        ['tanggal_kirim',  'datetime', 'Waktu absensi',               'Wajib'],
        ['jenis',          'M / P',    'Masuk atau Pulang',           'Wajib'],
        ['status_presensi','T2..PC4',  'Kode status dari mesin absen','Wajib'],
        ['dist_km',        'float',    'Jarak ke kantor (km)',        'Opsional'],
        ['approver_status','TERIMA/TOLAK','Keputusan atasan',         'Opsional'],
    ], columns=['Kolom', 'Tipe', 'Keterangan', 'Status']),
        use_container_width=True, hide_index=True)

# ============================================================
# UPLOAD
# ============================================================

def page_upload(uploaded):
    st.markdown("## 📥 Upload Data Absensi")
    local_files = scan_local_files()
    if local_files:
        st.success(f"📂 **{len(local_files)} file** ditemukan di direktori.")
        cs, cl = st.columns([4, 1])
        with cs:
            chosen = st.selectbox("Pilih file lokal", local_files, key='lf')
        with cl:
            st.markdown("<br>", unsafe_allow_html=True)
            load_clicked = st.button("📂 Load", type="primary", use_container_width=True, key='bl')
        # _finalize dipanggil di luar blok columns agar tampil full-width
        if load_clicked:
            with st.spinner(f"⏳ Memuat {chosen}..."):
                df, fc, remaps = load_local_file(chosen)
            _finalize(df, fc, remaps, chosen)
            return

    if uploaded is None:
        st.info("💡 Upload file via sidebar atau pilih file lokal di atas.")
        return

    with st.spinner("⏳ Memuat file..."):
        df, fc, remaps = load_processed_file(uploaded.getvalue(), uploaded.name)
    _finalize(df, fc, remaps, uploaded.name)


def _finalize(df, fixed_cols, remaps, source):
    st.success(f"✅ **{source}** — **{len(df):,} baris**, **{len(df.columns)} kolom**")

    if fixed_cols:
        st.info(f"🔧 Auto-fix desimal: `{'`, `'.join(fixed_cols)}`")

    for w in validate_dataframe(df):
        st.warning(f"⚠️ {w}")

    # Tampilkan remap info
    if remaps:
        remap_html = "".join(
            f"<div class='alert-box alert-blue'>{r}</div>" for r in remaps)
        st.markdown(remap_html, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Baris", f"{len(df):,}")
    with c2: st.metric("Karyawan Unik", f"{df['karyawan_id'].nunique():,}")
    with c3: st.metric("SKPD", f"{df['id_skpd'].nunique():,}" if 'id_skpd' in df.columns else '-')

    # Status distribution
    st.markdown("#### 📋 Distribusi Status Presensi")
    vc = df['status_presensi'].value_counts()
    n_cols = min(len(vc), 5)
    if n_cols > 0:
        cols = st.columns(n_cols)
        for i, (s, c) in enumerate(vc.items()):
            with cols[i % n_cols]:
                st.metric(f"{status_emoji(s)} {s}", f"{c:,}")

    # Cek apakah ada kode yang tidak dikenali
    unknown_set = set(df['status_presensi'].unique()) - STATUS_VALID - {'UNKNOWN'}
    if unknown_set:
        st.warning(f"⚠️ Label tidak dikenali (ditampilkan apa adanya): `{unknown_set}`")

    nb = df['is_bermasalah'].sum()
    if nb > 0:
        st.markdown(f"""<div class='alert-box alert-red'>
        🚨 <b>{nb:,}</b> absensi bermasalah (TELAT_BERAT / TELAT_SEDANG / PULANG_CEPAT_SEDANG / PULANG_CEPAT_BERAT)
        </div>""", unsafe_allow_html=True)

    if 'approver_status' in df.columns:
        st.markdown("#### 📋 Status Approver")
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1: st.metric("✅ TERIMA",           f"{df['is_terima'].sum():,}")
        with ac2: st.metric("❌ TOLAK",             f"{df['is_tolak'].sum():,}")
        with ac3: st.metric("⏳ PENDING",           f"{df['is_pending'].sum():,}")
        with ac4: st.metric("🚨 Terima Bermasalah", f"{df['terima_bermasalah'].sum():,}")

    with st.expander("🔍 Preview Data (10 baris)", expanded=False):
        cols_preview = [c for c in ['karyawan_id', 'id_skpd', 'jenis', 'tanggal_kirim',
                                     'status_presensi', 'dist_km', 'approver_status']
                        if c in df.columns]
        st.dataframe(df[cols_preview].head(10), use_container_width=True)

    if st.button("✅ Gunakan Data Ini", type="primary", use_container_width=True):
        st.session_state['df'] = df
        st.session_state['office_centroid'] = build_office_centroid(df)
        st.session_state['file_name'] = source
        st.rerun()

# ============================================================
# MAP HELPERS
# ============================================================

def build_popup(row):
    s  = row.get('status_presensi', '-')
    c  = status_color(s)
    e  = status_emoji(s)
    d  = row.get('dist_km', 0)
    tb = row.get('terima_bermasalah', 0)
    tb_h = ("<tr><td colspan=2><b style='color:#c0392b'>⚠️ TERIMA BERMASALAH!</b></td></tr>"
            if tb else "")
    return f"""<div style='font-family:Arial;font-size:12px;min-width:240px'>
      <h4 style='margin:0 0 8px;color:#2c3e50'>📋 Detail</h4>
      <table style='width:100%;border-collapse:collapse'>
        <tr><td><b>Karyawan</b></td><td>{row.get('karyawan_id','')}</td></tr>
        <tr><td><b>SKPD</b></td><td>{row.get('id_skpd','')}</td></tr>
        <tr><td><b>Jenis</b></td><td>{'🟢 Masuk' if row.get('jenis')=='M' else '🔴 Pulang'}</td></tr>
        <tr><td><b>Waktu</b></td><td>{str(row.get('tanggal_kirim',''))[:16]}</td></tr>
        <tr><td><b>Status</b></td><td>{e} <b style='color:{c}'>{s}</b></td></tr>
        <tr><td><b>Jarak</b></td><td>{d:.3f} km</td></tr>
        <tr><td><b>Approver</b></td><td>{row.get('approver_status','-') or '-'}</td></tr>
        {tb_h}
      </table>
    </div>"""


def create_folium_map(df, map_type='marker', oc=None):
    m = folium.Map(location=[df['lat'].median(), df['long'].median()],
                   zoom_start=13, tiles='CartoDB positron')
    folium.TileLayer('OpenStreetMap', name='OSM').add_to(m)

    if map_type == 'heatmap':
        HeatMap([[r['lat'], r['long'], 1 + r.get('is_bermasalah', 0)*3]
                 for _, r in df.iterrows()], radius=15, blur=10).add_to(m)
    elif map_type == 'cluster':
        mc = MarkerCluster(name='Absensi').add_to(m)
        for _, row in df.iterrows():
            fc = status_folium_color(row.get('status_presensi', ''))
            folium.CircleMarker([row['lat'], row['long']], radius=7,
                color=fc, fill=True, fill_color=fc, fill_opacity=0.75,
                popup=folium.Popup(build_popup(row), max_width=280)).add_to(mc)
    else:
        present = df['status_presensi'].unique()
        for s in STATUS_ORDER:
            if s not in present: continue
            fg = folium.FeatureGroup(name=f'{status_emoji(s)} {s}')
            fc = status_folium_color(s)
            berm = is_bermasalah(s)
            for _, row in df[df['status_presensi'] == s].iterrows():
                ec = 'gold' if row.get('terima_bermasalah', 0) == 1 else fc
                folium.CircleMarker([row['lat'], row['long']],
                    radius=9 if berm else 6,
                    color=ec, fill=True, fill_color=fc, fill_opacity=0.75,
                    popup=folium.Popup(build_popup(row), max_width=280)).add_to(fg)
            fg.add_to(m)
        unk = df[~df['status_presensi'].isin(STATUS_ORDER)]
        if not unk.empty:
            fg = folium.FeatureGroup(name='⚪ Lainnya')
            for _, row in unk.iterrows():
                folium.CircleMarker([row['lat'], row['long']], radius=5,
                    color='gray', fill=True, fill_color='gray', fill_opacity=0.5,
                    popup=folium.Popup(build_popup(row), max_width=280)).add_to(fg)
            fg.add_to(m)

    if oc is not None and len(oc):
        fg_o = folium.FeatureGroup(name='🏢 Kantor')
        for _, o in oc.iterrows():
            if pd.notna(o.get('office_lat')):
                folium.Marker([o['office_lat'], o['office_long']],
                    popup=f"SKPD {o['id_skpd']}",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa')).add_to(fg_o)
                folium.Circle([o['office_lat'], o['office_long']], radius=300,
                    color='#3498db', fill=False, weight=2, dash_array='5').add_to(fg_o)
        fg_o.add_to(m)

    folium.LayerControl().add_to(m)
    return m

# ============================================================
# VISUALISASI
# ============================================================

def page_visualisasi(filters):
    st.markdown("## 📊 Visualisasi Absensi")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Upload data dulu.")
        return
    df_full = st.session_state.df
    oc = st.session_state.get('office_centroid')
    h = _df_hash(df_full)
    df = apply_filters(h, df_full,
        filters.get('skpd', 'Semua'),
        tuple(filters.get('jenis', ['M', 'P'])),
        filters.get('date'),
        filters.get('dist', (0.0, 100.0)),
        filters.get('approver', 'Semua'),
        filters.get('status'))
    st.caption(f"📊 **{len(df):,}** dari **{len(df_full):,}** baris")
    tabs = st.tabs(["📊 Overview", "🗺️ Peta", "⏰ Temporal",
                    "📏 Jarak", "👤 Karyawan", "📋 Approver", "📋 Data"])
    with tabs[0]: _vis_overview(df)
    with tabs[1]: _vis_map(df, filters, oc)
    with tabs[2]: _vis_temporal(df)
    with tabs[3]: _vis_distance(df)
    with tabs[4]: _vis_employee(df)
    with tabs[5]: _vis_approver(df)
    with tabs[6]: _vis_data(df)


def _vis_overview(df):
    n = len(df); nb = df['is_bermasalah'].sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total", f"{n:,}")
    with c2: st.metric("Karyawan", f"{df['karyawan_id'].nunique():,}")
    with c3: st.metric("🔴🟠 Bermasalah", f"{nb:,}")
    with c4: st.metric("🟢🟡 OK", f"{n - nb:,}")
    if df['terima_bermasalah'].sum() > 0:
        st.markdown(f"""<div class='alert-box alert-red'>
        🚨 <b>{df['terima_bermasalah'].sum():,} Terima Bermasalah</b>
        </div>""", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        vc = df['status_presensi'].value_counts().reset_index()
        vc.columns = ['status_presensi', 'count']
        fig = px.pie(vc, values='count', names='status_presensi',
                     title='Distribusi Status Presensi',
                     color='status_presensi', color_discrete_map=STATUS_COLORS, hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        skpd_s = df.groupby(['id_skpd', 'status_presensi']).size().reset_index(name='n')
        fig = px.bar(skpd_s, x='id_skpd', y='n', color='status_presensi',
                     title='Status per SKPD', barmode='stack',
                     color_discrete_map=STATUS_COLORS,
                     category_orders={'status_presensi': STATUS_ORDER})
        fig.update_xaxes(type='category')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    cl2, cr2 = st.columns(2)
    with cl2:
        masuk = df[df['jenis'] == 'M']
        if not masuk.empty:
            vm = masuk['status_presensi'].value_counts().reset_index()
            vm.columns = ['status_presensi', 'count']
            fig = px.pie(vm, values='count', names='status_presensi', title='🟢 MASUK',
                         color='status_presensi', color_discrete_map=STATUS_COLORS, hole=0.4)
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
    with cr2:
        pulang = df[df['jenis'] == 'P']
        if not pulang.empty:
            vp = pulang['status_presensi'].value_counts().reset_index()
            vp.columns = ['status_presensi', 'count']
            fig = px.pie(vp, values='count', names='status_presensi', title='🔴 PULANG',
                         color='status_presensi', color_discrete_map=STATUS_COLORS, hole=0.4)
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)


def _vis_map(df, filters, oc):
    if df.empty: st.warning("Tidak ada data."); return
    MAX = 2000
    dfd = df.sample(MAX, random_state=42) if len(df) > MAX else df
    if len(df) > MAX: st.info(f"Menampilkan {MAX:,} dari {len(df):,} titik.")
    m = create_folium_map(dfd, filters.get('map_type', 'marker'), oc)
    st_folium(m, width=None, height=560, returned_objects=[])


def _vis_temporal(df):
    if 'jam' not in df.columns: st.warning("Kolom jam tidak ada."); return
    cl, cr = st.columns(2)
    with cl:
        fig = px.bar(df.groupby(['jam','status_presensi']).size().reset_index(name='n'),
                     x='jam', y='n', color='status_presensi', title='Status per Jam',
                     color_discrete_map=STATUS_COLORS, category_orders={'status_presensi':STATUS_ORDER})
        fig.add_vrect(x0=7,x1=9,fillcolor='green',opacity=0.07,annotation_text='Masuk')
        fig.add_vrect(x0=15,x1=17,fillcolor='purple',opacity=0.07,annotation_text='Pulang')
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        if 'weekday' in df.columns:
            dm = {0:'Senin',1:'Selasa',2:'Rabu',3:'Kamis',4:'Jumat',5:'Sabtu',6:'Minggu'}
            d2 = df.copy(); d2['hari'] = d2['weekday'].map(dm)
            fig = px.bar(d2.groupby(['hari','status_presensi']).size().reset_index(name='n'),
                         x='hari', y='n', color='status_presensi', title='Status per Hari',
                         color_discrete_map=STATUS_COLORS,
                         category_orders={'hari':list(dm.values()),'status_presensi':STATUS_ORDER})
            st.plotly_chart(fig, use_container_width=True)
    if 'tanggal' in df.columns:
        daily = df.groupby(['tanggal','status_presensi']).size().reset_index(name='n')
        fig = px.line(daily, x='tanggal', y='n', color='status_presensi',
                      markers=True, title='Trend Harian', color_discrete_map=STATUS_COLORS)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)


def _vis_distance(df):
    if 'dist_km' not in df.columns: st.warning("Kolom dist_km tidak ada."); return
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Rata-rata", f"{df['dist_km'].mean():.3f} km")
    with c2: st.metric("Median",    f"{df['dist_km'].median():.3f} km")
    with c3: st.metric("Maks",      f"{df['dist_km'].max():.3f} km")
    with c4: st.metric(">300m",     f"{(df['dist_km']>0.3).sum():,}")
    cl, cr = st.columns(2)
    with cl:
        fig = px.histogram(df[df['dist_km']<=10], x='dist_km',
                           title='Distribusi Jarak ≤10km', nbins=50,
                           color_discrete_sequence=['#3498db'])
        fig.add_vline(x=0.3, line_dash='dash', line_color='red',
                      annotation_text='300m', annotation_position='top right')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        # Box plot jarak per SKPD — tanpa breakdown status
        skpd_dist = df[df['dist_km']<=10].groupby('id_skpd')['dist_km'].median().reset_index()
        skpd_dist = skpd_dist.sort_values('dist_km', ascending=False)
        fig = px.bar(skpd_dist, x='id_skpd', y='dist_km',
                     title='Median Jarak per SKPD',
                     color='dist_km', color_continuous_scale='Blues_r')
        fig.add_hline(y=0.3, line_dash='dash', line_color='red',
                      annotation_text='300m')
        fig.update_xaxes(type='category')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    # Row kedua: distribusi outside/inside 300m
    cl2, cr2 = st.columns(2)
    with cl2:
        zone = pd.DataFrame({
            'Zona': ['Dalam 300m', 'Di luar 300m'],
            'Jumlah': [(df['dist_km']<=0.3).sum(), (df['dist_km']>0.3).sum()]
        })
        fig = px.pie(zone, values='Jumlah', names='Zona',
                     title='Proporsi Dalam vs Luar 300m',
                     color='Zona',
                     color_discrete_map={'Dalam 300m':'#27ae60','Di luar 300m':'#e74c3c'},
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with cr2:
        # Histogram zoom untuk jarak 0-500m
        fig = px.histogram(df[df['dist_km']<=0.5], x='dist_km',
                           title='Distribusi Jarak ≤500m (zoom)', nbins=50,
                           color_discrete_sequence=['#2ecc71'])
        fig.add_vline(x=0.3, line_dash='dash', line_color='red',
                      annotation_text='300m')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def _vis_employee(df):
    pivot = df.groupby(['karyawan_id','status_presensi']).size().unstack(fill_value=0)
    pivot['total'] = pivot.sum(axis=1)
    pivot['bermasalah_n'] = sum(pivot.get(s, 0) for s in STATUS_BERMASALAH)
    pivot['bermasalah_pct'] = (pivot['bermasalah_n'] / pivot['total'] * 100).round(1)
    pivot['skpd'] = df.groupby('karyawan_id')['id_skpd'].first()
    pivot = pivot.reset_index().sort_values('bermasalah_n', ascending=False)
    cl, cr = st.columns(2)
    with cl:
        top = pivot.nlargest(15, 'bermasalah_n')
        berm_cols = [s for s in STATUS_BERMASALAH if s in top.columns]
        if berm_cols:
            fig = px.bar(top, x='karyawan_id', y=berm_cols, title='Top 15 Bermasalah',
                         color_discrete_map=STATUS_COLORS, barmode='stack')
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig = px.scatter(pivot, x='total', y='bermasalah_pct', size='bermasalah_n',
                         color='bermasalah_pct', hover_data=['karyawan_id','skpd'],
                         title='Total vs % Bermasalah', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    risky = pivot[pivot['bermasalah_n'] > 0]
    if len(risky):
        st.markdown("### 🚨 Karyawan Bermasalah")
        show_cols = ['karyawan_id','skpd','total','bermasalah_n','bermasalah_pct'] + \
                    [s for s in STATUS_ORDER if s in risky.columns]
        st.dataframe(risky[show_cols].head(30), use_container_width=True)


def _vis_approver(df):
    st.markdown("### 📋 Analisis Approver")
    if 'approver_status' not in df.columns: st.info("Kolom approver_status tidak ada."); return

    # ── Metric ringkasan ──────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("✅ TERIMA",            f"{df['is_terima'].sum():,}")
    with c2: st.metric("❌ TOLAK",             f"{df['is_tolak'].sum():,}")
    with c3: st.metric("⏳ PENDING",           f"{df['is_pending'].sum():,}")
    with c4: st.metric("🚨 Terima Bermasalah", f"{df['terima_bermasalah'].sum():,}")

    # ── Chart status vs approver ──────────────────────────────
    cross = df.groupby(['status_presensi','approver_status']).size().reset_index(name='n')
    cross = cross[cross['approver_status'] != '']
    if not cross.empty:
        fig = px.bar(cross, x='status_presensi', y='n', color='approver_status',
                     title='Status vs Approver', barmode='group',
                     category_orders={'status_presensi':STATUS_ORDER})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    base_cols = ['karyawan_id','id_skpd','jenis','tanggal_kirim','status_presensi',
                 'dist_km','approver_status','catatan','anomaly_score','risk_level',
                 'is_noise_masuk','is_noise_pulang','is_st_noise_masuk','is_st_noise_pulang']

    tabs_approver = st.tabs(["🚨 Terima Bermasalah", "📍 Noise di-TERIMA", "⏳ Pending"])

    # ── Tab 1: Terima Bermasalah ──────────────────────────────
    with tabs_approver[0]:
        tb = df[df['terima_bermasalah'] == 1]
        if len(tb):
            st.markdown(f"""<div class='alert-box alert-red'>
            🚨 <b>{len(tb):,} absensi bermasalah (TELAT_BERAT/SEDANG atau PULANG_CEPAT_BERAT/SEDANG) tetap di-TERIMA approver</b>
            </div>""", unsafe_allow_html=True)
            cols = [c for c in base_cols if c in tb.columns]
            st.dataframe(tb[cols].sort_values('anomaly_score', ascending=False)
                         if 'anomaly_score' in tb.columns else tb[cols],
                         use_container_width=True, height=400)
            st.download_button("⬇️ Download Terima Bermasalah",
                               tb[cols].to_csv(index=False).encode(),
                               "terima_bermasalah.csv", "text/csv")
        else:
            st.success("✅ Tidak ada absensi bermasalah yang di-TERIMA.")

    # ── Tab 2: Noise DBSCAN / ST-DBSCAN yang di-TERIMA ───────
    with tabs_approver[1]:
        noise_cols = [c for c in ['is_noise_masuk','is_noise_pulang',
                                   'is_st_noise_masuk','is_st_noise_pulang'] if c in df.columns]
        if not noise_cols:
            st.info("Kolom noise (DBSCAN/ST-DBSCAN) tidak tersedia. Jalankan preprocessing terlebih dahulu.")
        else:
            # Baris yang merupakan noise (outlier spasial/spatio-temporal) DAN di-TERIMA
            noise_mask = df[noise_cols].any(axis=1)
            noise_terima = df[noise_mask & (df['is_terima'] == 1)]
            noise_any    = df[noise_mask]

            na1, na2, na3 = st.columns(3)
            with na1: st.metric("Total noise (DBSCAN/ST)", f"{noise_mask.sum():,}")
            with na2: st.metric("Noise di-TERIMA",         f"{len(noise_terima):,}")
            with na3: st.metric("Noise di-TOLAK",          f"{(noise_mask & (df['is_tolak']==1)).sum():,}")

            # Breakdown per jenis noise
            st.markdown("**Rincian noise per tipe:**")
            nc1, nc2, nc3, nc4 = st.columns(4)
            with nc1: st.metric("Noise Masuk",    f"{df.get('is_noise_masuk', pd.Series(0)).sum():,}")
            with nc2: st.metric("Noise Pulang",   f"{df.get('is_noise_pulang', pd.Series(0)).sum():,}")
            with nc3: st.metric("ST-Noise Masuk", f"{df.get('is_st_noise_masuk', pd.Series(0)).sum():,}")
            with nc4: st.metric("ST-Noise Pulang",f"{df.get('is_st_noise_pulang', pd.Series(0)).sum():,}")

            if len(noise_terima):
                st.markdown(f"""<div class='alert-box alert-orange'>
                ⚠️ <b>{len(noise_terima):,} absensi outlier lokasi/waktu tetap di-TERIMA approver</b>
                — lokasi tidak konsisten dengan pola absensi normal karyawan tersebut.
                </div>""", unsafe_allow_html=True)
                cols = [c for c in base_cols + noise_cols if c in noise_terima.columns]
                st.dataframe(noise_terima[cols].sort_values('dist_km', ascending=False)
                             if 'dist_km' in noise_terima.columns else noise_terima[cols],
                             use_container_width=True, height=400)
                st.download_button("⬇️ Download Noise Terima",
                                   noise_terima[cols].to_csv(index=False).encode(),
                                   "noise_terima.csv", "text/csv")
            else:
                st.success("✅ Tidak ada noise yang di-TERIMA.")

    # ── Tab 3: PENDING ────────────────────────────────────────
    with tabs_approver[2]:
        pending = df[df['is_pending'] == 1]
        if len(pending):
            st.markdown(f"""<div class='alert-box alert-blue'>
            ⏳ <b>{len(pending):,} absensi belum diproses approver</b>
            </div>""", unsafe_allow_html=True)

            # Ringkasan pending per status presensi
            p_vc = pending['status_presensi'].value_counts().reset_index()
            p_vc.columns = ['status_presensi', 'jumlah']
            p_vc['prioritas'] = p_vc['status_presensi'].apply(
                lambda s: '🔴 Segera' if s in STATUS_BERMASALAH else '🟢 Normal')

            pa1, pa2 = st.columns(2)
            with pa1:
                fig = px.bar(p_vc, x='status_presensi', y='jumlah',
                             color='prioritas', title='Pending per Status',
                             color_discrete_map={'🔴 Segera':'#e74c3c','🟢 Normal':'#2ecc71'})
                fig.update_xaxes(type='category')
                st.plotly_chart(fig, use_container_width=True)
            with pa2:
                # Top karyawan dengan pending terbanyak
                top_pending = (pending.groupby('karyawan_id')
                               .agg(n_pending=('karyawan_id','count'),
                                    n_bermasalah=('is_bermasalah','sum'))
                               .reset_index()
                               .sort_values('n_bermasalah', ascending=False)
                               .head(15))
                fig = px.bar(top_pending, x='karyawan_id', y='n_pending',
                             color='n_bermasalah', title='Top 15 Karyawan — Pending Terbanyak',
                             color_continuous_scale='Reds')
                fig.update_xaxes(type='category')
                st.plotly_chart(fig, use_container_width=True)

            # Tabel detail pending, prioritaskan yang bermasalah dulu
            st.markdown("**Detail — bermasalah diprioritaskan di atas:**")
            cols = [c for c in base_cols if c in pending.columns]
            pending_sorted = pending.sort_values(['is_bermasalah','dist_km'],
                                                  ascending=[False,False])
            st.dataframe(pending_sorted[cols].head(200),
                         use_container_width=True, height=420)
            st.download_button("⬇️ Download Pending",
                               pending_sorted[cols].to_csv(index=False).encode(),
                               "pending_approver.csv", "text/csv")
        else:
            st.success("✅ Tidak ada absensi pending.")


def _vis_data(df):
    c1, c2 = st.columns(2)
    with c1: search = st.text_input("🔍 Cari Karyawan ID", "")
    with c2:
        sort_col = st.selectbox("Urutkan",
            [c for c in ['tanggal_kirim','dist_km','status_presensi'] if c in df.columns])
    dft = df.copy()
    if search: dft = dft[dft['karyawan_id'].astype(str).str.contains(search)]
    if sort_col in dft.columns:
        dft = dft.sort_values(sort_col, ascending=(sort_col == 'tanggal_kirim'))
    cols = [c for c in ['karyawan_id','id_skpd','jenis','tanggal_kirim',
                        'status_presensi','dist_km','approver_status',
                        'terima_bermasalah','catatan'] if c in dft.columns]
    st.dataframe(dft[cols].head(500), use_container_width=True, height=480)
    st.caption(f"{min(500,len(dft))} dari {len(dft):,}")
    st.download_button("⬇️ CSV", dft[cols].to_csv(index=False).encode(), "filtered.csv", "text/csv")

# ============================================================
# HUNTING
# ============================================================

def page_hunting():
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Upload data dulu.")
        return
    df = st.session_state.df
    oc = st.session_state.get('office_centroid', pd.DataFrame())
    st.markdown("""<div class="hunt-header">
        <div class="hunt-title">[ HUNTING MODE ]</div>
        <div class="hunt-sub">Investigasi mendalam — per status presensi</div>
    </div>""", unsafe_allow_html=True)
    n = len(df); nb = df['is_bermasalah'].sum(); tb = df['terima_bermasalah'].sum()
    vc = df['status_presensi'].value_counts()
    parts = []
    for s in STATUS_ORDER:
        if s in vc.index:
            parts.append(f"<span>{status_emoji(s)} <b style='color:{status_color(s)}'>"
                         f"{vc[s]:,}</b> {s}</span>")
    st.markdown(f"""<div style="background:#f0f2f6;border-radius:8px;padding:.6rem 1.2rem;
        margin-bottom:1rem;display:flex;gap:1rem;align-items:center;
        font-size:.8rem;flex-wrap:wrap">
        <span>📊 <b>{n:,}</b></span>{' '.join(parts)}
        <span>👤 <b>{df['karyawan_id'].nunique():,}</b></span>
        {"<span>🚨 <b style='color:#c0392b'>" + str(tb) + "</b> Terima Brmslh</span>" if tb else ""}
    </div>""", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["🕵️ By Pegawai", "🏢 By SKPD", "📅 By Tanggal"])
    with t1: _hunt_pegawai(df, oc)
    with t2: _hunt_skpd(df, oc)
    with t3: _hunt_tanggal(df, oc)


def _hunt_pegawai(df, oc):
    st.markdown("""<div class="section-header"><span style="font-size:1.5rem">🕵️</span>
        <div><div style="font-size:1.1rem;font-weight:700;color:#2c3e50">Hunt by Pegawai</div>
        <div style="font-size:.78rem;color:#7f8c8d">Timeline, jejak, riwayat lengkap</div></div>
    </div>""", unsafe_allow_html=True)
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = []
    ids = sorted(df['karyawan_id'].unique().tolist())
    cs, cb = st.columns([4, 1])
    with cs:
        sel = st.selectbox("🔎 Pilih Pegawai", ids,
            format_func=lambda x: (
                f"ID {x} | SKPD {df[df['karyawan_id']==x]['id_skpd'].iloc[0] if len(df[df['karyawan_id']==x]) else '-'}"
                f" | Bermasalah: {df[df['karyawan_id']==x]['is_bermasalah'].sum()}"),
            key='hp_id')
    with cb:
        st.markdown("<br>", unsafe_allow_html=True)
        in_wl = sel in st.session_state['watchlist']
        if st.button("📌 Watch" if not in_wl else "❌ Unwatch", use_container_width=True):
            if in_wl: st.session_state['watchlist'].remove(sel)
            else: st.session_state['watchlist'].append(sel)
            st.rerun()
    de = df[df['karyawan_id'] == sel].sort_values('tanggal_kirim')
    if de.empty: st.warning("Tidak ada data."); return
    tot = len(de); nb = de['is_bermasalah'].sum()
    skpd_e = de['id_skpd'].mode()[0]
    avg_km = de['dist_km'].mean() if 'dist_km' in de.columns else 0
    n_tb = de['terima_bermasalah'].sum()
    st.markdown(f"""<div class="metric-grid">
        <div class="metric-card mc-blue"><div class="metric-val">{tot}</div><div class="metric-lbl">Total</div></div>
        <div class="metric-card mc-red"><div class="metric-val">{nb}</div><div class="metric-lbl">Bermasalah</div></div>
        <div class="metric-card mc-green"><div class="metric-val">{tot-nb}</div><div class="metric-lbl">OK</div></div>
        <div class="metric-card"><div class="metric-val">{avg_km:.3f} km</div><div class="metric-lbl">Avg Jarak</div></div>
        <div class="metric-card {'mc-red' if n_tb else ''}"><div class="metric-val">{n_tb}</div><div class="metric-lbl">Terima Brmslh</div></div>
    </div>""", unsafe_allow_html=True)
    vc2 = de['status_presensi'].value_counts()
    n_sc = min(len(vc2), 5)
    if n_sc > 0:
        sc_cols = st.columns(n_sc)
        for i, (s, c) in enumerate(vc2.items()):
            with sc_cols[i % n_sc]:
                st.metric(f"{status_emoji(s)} {s}", c)
    t1, t2, t3, t4, t5 = st.tabs(["📅 Timeline","🗺️ Jejak","📊 Vs SKPD","📋 Approver","📋 Riwayat"])
    with t1:
        if 'tanggal' in de.columns and 'jam_desimal' in de.columns:
            dp = de.copy(); dp['ukuran'] = dp['is_bermasalah'] * 8 + 4
            hover = ['status_presensi'] + (['dist_km'] if 'dist_km' in dp.columns else [])
            fig = px.scatter(dp, x='tanggal', y='jam_desimal',
                             color='status_presensi', symbol='jenis', size='ukuran',
                             color_discrete_map=STATUS_COLORS,
                             title=f'Timeline — ID {sel}', hover_data=hover)
            fig.add_hline(y=7.5, line_dash='dot', line_color='#3498db', annotation_text='07:30')
            fig.add_hline(y=16.0, line_dash='dot', line_color='#9b59b6', annotation_text='16:00')
            fig.update_layout(height=420, plot_bgcolor='#fafafa')
            st.plotly_chart(fig, use_container_width=True)
    with t2:
        ctr = [de['lat'].median(), de['long'].median()]
        mp = folium.Map(location=ctr, zoom_start=14, tiles='CartoDB positron')
        coords = [[r['lat'], r['long']] for _, r in de.iterrows()]
        if len(coords) > 1:
            AntPath(locations=coords, color='#667eea', weight=2.5, opacity=0.6,
                    delay=800, dash_array=[10, 20]).add_to(mp)
        for i, (_, row) in enumerate(de.iterrows()):
            fc = status_folium_color(row.get('status_presensi',''))
            border = 'gold' if row.get('terima_bermasalah',0) == 1 else fc
            berm = is_bermasalah(row.get('status_presensi',''))
            popup = (f"<div style='font-size:12px'><b>#{i+1} — {str(row.get('tanggal_kirim',''))[:16]}</b>"
                     f"<br>{'🟢 Masuk' if row.get('jenis')=='M' else '🔴 Pulang'}"
                     f"<br>Status: <b>{row.get('status_presensi','')}</b>"
                     f"<br>Jarak: {row.get('dist_km',0):.3f} km</div>")
            folium.CircleMarker([row['lat'], row['long']], radius=11 if berm else 7,
                color=border, fill=True, fill_color=fc, fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=230)).add_to(mp)
        if not oc.empty:
            off = oc[oc['id_skpd'] == skpd_e]
            if not off.empty:
                o = off.iloc[0]
                folium.Marker([o['office_lat'], o['office_long']], popup=f"Kantor {skpd_e}",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa')).add_to(mp)
                folium.Circle([o['office_lat'], o['office_long']], radius=300,
                    color='#3498db', fill=False, weight=2, dash_array='5').add_to(mp)
        st_folium(mp, width=None, height=500, returned_objects=[])
    with t3:
        df_s = df[df['id_skpd'] == skpd_e]
        ag = df_s.groupby('karyawan_id').agg(
            bermasalah_n=('is_bermasalah','sum'),
            total=('karyawan_id','count')).reset_index()
        ag['pct'] = (ag['bermasalah_n'] / ag['total'] * 100).round(1)
        emp_r = ag[ag['karyawan_id'] == sel]
        if not emp_r.empty:
            e = emp_r.iloc[0]
            rank = (ag['bermasalah_n'] > e['bermasalah_n']).sum() + 1
            st.markdown(f"#### SKPD {skpd_e} — Peringkat **{rank}** dari {len(ag)}")
        fig = px.scatter(ag, x='total', y='pct', size='bermasalah_n', color='pct',
                         color_continuous_scale='RdYlGn_r', hover_data=['karyawan_id'],
                         title=f'Sebaran SKPD {skpd_e}')
        if not emp_r.empty:
            e = emp_r.iloc[0]
            fig.add_annotation(x=e['total'], y=e['pct'], text=f"▶ ID {sel}", showarrow=True,
                               arrowhead=2, font=dict(color='#c0392b', size=12))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with t4:
        if 'approver_status' in de.columns:
            cross = de.groupby(['status_presensi','approver_status']).size().reset_index(name='n')
            cross = cross[cross['approver_status'] != '']
            if not cross.empty:
                fig = px.bar(cross, x='status_presensi', y='n', color='approver_status',
                             title='Status vs Approver', barmode='group')
                fig.update_layout(height=340)
                st.plotly_chart(fig, use_container_width=True)
            tb_r = de[de['terima_bermasalah'] == 1]
            if len(tb_r):
                st.error(f"🚨 {len(tb_r)} Terima Bermasalah!")
                cols = [c for c in ['tanggal_kirim','jenis','status_presensi',
                                    'dist_km','approver_status'] if c in tb_r.columns]
                st.dataframe(tb_r[cols], use_container_width=True)
    with t5:
        cols = [c for c in ['tanggal_kirim','jenis','status_presensi',
                            'dist_km','approver_status','catatan'] if c in de.columns]
        st.dataframe(de[cols].sort_values('tanggal_kirim', ascending=False),
                     use_container_width=True, height=400)
        st.download_button(f"⬇️ ID {sel}", de[cols].to_csv(index=False).encode(),
                           f"karyawan_{sel}.csv", "text/csv")


def _hunt_skpd(df, oc):
    st.markdown("""<div class="section-header"><span style="font-size:1.5rem">🏢</span>
        <div><div style="font-size:1.1rem;font-weight:700;color:#2c3e50">Hunt by SKPD</div></div>
    </div>""", unsafe_allow_html=True)
    skpds = sorted(df['id_skpd'].unique().tolist())
    sel_s = st.selectbox("🏢 SKPD", skpds,
        format_func=lambda x: f"SKPD {x} ({len(df[df['id_skpd']==x]):,} absensi)",
        key='hs_id')
    ds = df[df['id_skpd'] == sel_s].copy()
    if ds.empty: st.warning("Tidak ada data."); return
    nk = ds['karyawan_id'].nunique(); nb = ds['is_bermasalah'].sum(); n_tb = ds['terima_bermasalah'].sum()
    st.markdown(f"""<div class="metric-grid">
        <div class="metric-card mc-blue"><div class="metric-val">{len(ds):,}</div><div class="metric-lbl">Total</div></div>
        <div class="metric-card mc-blue"><div class="metric-val">{nk}</div><div class="metric-lbl">Karyawan</div></div>
        <div class="metric-card mc-red"><div class="metric-val">{nb:,}</div><div class="metric-lbl">Bermasalah</div></div>
        <div class="metric-card"><div class="metric-val">{nb/max(len(ds),1)*100:.1f}%</div><div class="metric-lbl">%</div></div>
        <div class="metric-card {'mc-red' if n_tb else ''}"><div class="metric-val">{n_tb}</div><div class="metric-lbl">Terima Brmslh</div></div>
    </div>""", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["🏆 Leaderboard","🔥 Heatmap","📅 Trend","📋 Approver"])
    with t1:
        pv = ds.groupby(['karyawan_id','status_presensi']).size().unstack(fill_value=0)
        pv['total'] = pv.sum(axis=1)
        pv['bermasalah_n'] = sum(pv.get(s, 0) for s in STATUS_BERMASALAH)
        pv['pct'] = (pv['bermasalah_n'] / pv['total'] * 100).round(1)
        pv = pv.reset_index().sort_values('bermasalah_n', ascending=False)
        berm_cols = [s for s in STATUS_BERMASALAH if s in pv.columns]
        if berm_cols:
            fig = px.bar(pv.head(15), x='karyawan_id', y=berm_cols,
                         title=f'Top 15 SKPD {sel_s}',
                         color_discrete_map=STATUS_COLORS, barmode='stack')
            fig.update_xaxes(type='category'); fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pv.head(30), use_container_width=True, height=320)
    with t2:
        mp = folium.Map(location=[ds['lat'].median(), ds['long'].median()],
                        zoom_start=13, tiles='CartoDB positron')
        HeatMap([[r['lat'],r['long'],1+r.get('is_bermasalah',0)*3] for _,r in ds.iterrows()],
                radius=18, blur=12,
                gradient={'0.0':'green','0.5':'yellow','1.0':'red'}).add_to(mp)
        if not oc.empty:
            off = oc[oc['id_skpd'] == sel_s]
            if not off.empty:
                o = off.iloc[0]
                folium.Marker([o['office_lat'], o['office_long']], popup=f"Kantor {sel_s}",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa')).add_to(mp)
        st_folium(mp, width=None, height=500, returned_objects=[])
    with t3:
        if 'tanggal' in ds.columns:
            daily = ds.groupby(['tanggal','status_presensi']).size().reset_index(name='n')
            fig = px.area(daily, x='tanggal', y='n', color='status_presensi',
                          title='Trend', color_discrete_map=STATUS_COLORS,
                          category_orders={'status_presensi':STATUS_ORDER})
            fig.update_layout(height=380); st.plotly_chart(fig, use_container_width=True)
    with t4:
        if 'approver_status' in ds.columns:
            cross = ds.groupby(['status_presensi','approver_status']).size().reset_index(name='n')
            cross = cross[cross['approver_status'] != '']
            if not cross.empty:
                fig = px.bar(cross, x='status_presensi', y='n', color='approver_status',
                             title='Status vs Approver', barmode='group')
                fig.update_layout(height=380); st.plotly_chart(fig, use_container_width=True)
            if n_tb > 0:
                st.error(f"🚨 {n_tb} Terima Bermasalah!")


def _hunt_tanggal(df, oc):
    st.markdown("""<div class="section-header"><span style="font-size:1.5rem">📅</span>
        <div><div style="font-size:1.1rem;font-weight:700;color:#2c3e50">Hunt by Tanggal</div></div>
    </div>""", unsafe_allow_html=True)
    if 'tanggal' not in df.columns: st.warning("Kolom tanggal tidak ada."); return
    mn, mx = df['tanggal'].min(), df['tanggal'].max()
    c1, c2 = st.columns(2)
    with c1:
        dr = st.date_input("📅 Rentang", value=(mx - timedelta(days=6), mx),
                           min_value=mn, max_value=mx, key='hd_dr')
    with c2:
        fs = st.multiselect("Filter Status", STATUS_ORDER, default=STATUS_ORDER,
                            key='hd_fs', format_func=lambda x: f"{status_emoji(x)} {x}")
    d_s, d_e = (dr if isinstance(dr, tuple) and len(dr)==2 else (dr, dr))
    dd = df[(df['tanggal'].astype('datetime64[ns]') >= pd.Timestamp(d_s)) &
            (df['tanggal'].astype('datetime64[ns]') <= pd.Timestamp(d_e))].copy()
    if fs: dd = dd[dd['status_presensi'].isin(fs)]
    if dd.empty: st.warning("Tidak ada data."); return
    nb = dd['is_bermasalah'].sum()
    st.markdown(f"""<div class="metric-grid">
        <div class="metric-card mc-blue"><div class="metric-val">{len(dd):,}</div><div class="metric-lbl">Absensi</div></div>
        <div class="metric-card mc-blue"><div class="metric-val">{dd['karyawan_id'].nunique()}</div><div class="metric-lbl">Karyawan</div></div>
        <div class="metric-card mc-red"><div class="metric-val">{nb:,}</div><div class="metric-lbl">Bermasalah</div></div>
    </div>""", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["🗺️ Peta","⏰ Jam","🚨 Karyawan","🔍 Deteksi Titipan"])
    with t1:
        MAX = 1500
        disp = dd.sample(MAX, random_state=42) if len(dd) > MAX else dd
        mp = folium.Map(location=[disp['lat'].median(), disp['long'].median()],
                        zoom_start=13, tiles='CartoDB positron')
        mc = MarkerCluster(name='Absensi').add_to(mp)
        for _, row in disp.iterrows():
            fc = status_folium_color(row.get('status_presensi',''))
            folium.CircleMarker([row['lat'],row['long']], radius=7,
                color=fc, fill=True, fill_color=fc, fill_opacity=0.75,
                popup=folium.Popup(build_popup(row), max_width=230)).add_to(mc)
        folium.LayerControl().add_to(mp)
        st_folium(mp, width=None, height=500, returned_objects=[])
    with t2:
        if 'jam' in dd.columns:
            fig = px.bar(dd.groupby(['jam','status_presensi']).size().reset_index(name='n'),
                         x='jam', y='n', color='status_presensi', title='Status per Jam',
                         color_discrete_map=STATUS_COLORS, category_orders={'status_presensi':STATUS_ORDER})
            fig.add_vrect(x0=7,x1=9,fillcolor='green',opacity=0.07,annotation_text='Masuk')
            fig.add_vrect(x0=15,x1=17,fillcolor='purple',opacity=0.07,annotation_text='Pulang')
            st.plotly_chart(fig, use_container_width=True)
    with t3:
        emp = dd.groupby(['karyawan_id','id_skpd']).agg(
            n_abs=('karyawan_id','count'), n_berm=('is_bermasalah','sum')).reset_index()
        emp = emp.sort_values('n_berm', ascending=False)
        min_b = st.slider("Min bermasalah", 0, max(1, int(emp['n_berm'].max())), 0, key='hd_mb')
        st.dataframe(emp[emp['n_berm'] >= min_b], use_container_width=True, height=380)
    with t4:
        st.markdown("#### 🔍 Deteksi Titipan Absensi")
        cr1, cr2 = st.columns(2)
        with cr1: rad = st.slider("Radius (m)", 5, 200, 50, 5, key='hd_r')
        with cr2: win = st.slider("Jendela (mnt)", 1, 60, 15, 1, key='hd_w')
        n_ck = len(dd)
        if n_ck * (n_ck - 1) // 2 > 500000:
            st.warning("Terlalu banyak data. Perkecil rentang tanggal.")
        elif st.button("🔍 Deteksi Sekarang", type="primary", key='hd_det'):
            with st.spinner("⏳ Menganalisis..."):
                ck = dd[['karyawan_id','lat','long','tanggal_kirim']].dropna()
                rows_list = ck.to_dict('records')
                sus = []
                for i in range(len(rows_list)):
                    for j in range(i+1, len(rows_list)):
                        r1, r2 = rows_list[i], rows_list[j]
                        if r1['karyawan_id'] == r2['karyawan_id']: continue
                        dm = haversine(r1['lat'],r1['long'],r2['lat'],r2['long'])
                        if dm > rad: continue
                        dt = abs((pd.Timestamp(r1['tanggal_kirim']) -
                                  pd.Timestamp(r2['tanggal_kirim'])).total_seconds()) / 60
                        if dt <= win:
                            sus.append({'Karyawan A':r1['karyawan_id'],'Karyawan B':r2['karyawan_id'],
                                        'Jarak (m)':round(dm,1),'Selisih (mnt)':round(dt,1)})
                        if len(sus) >= 300: break
                    if len(sus) >= 300: break
                st.session_state['kolusi'] = sus
        if 'kolusi' in st.session_state:
            kol = st.session_state['kolusi']
            if not kol: st.success("✅ Tidak ditemukan pasangan mencurigakan.")
            else:
                st.error(f"🚨 {len(kol)} pasangan mencurigakan!")
                st.dataframe(pd.DataFrame(kol), use_container_width=True)

# ============================================================
# PREDIKSI
# ============================================================

def page_prediksi():
    st.markdown("## 🔮 Prediksi Status Presensi")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Upload data dulu.")
        return
    oc = st.session_state.get('office_centroid', pd.DataFrame())
    col_m, col_p = st.columns(2)
    with col_m:
        st.markdown("""### ⏰ MASUK
| Jam | Kode | Status |
|-----|------|--------|
| ≤ 07:30 | TWM | 🟢 TEPAT_WAKTU_MASUK |
| 07:31–08:00 | T2 | 🟡 TELAT_RINGAN |
| 08:01–09:00 | T3 | 🟠 TELAT_SEDANG |
| > 09:00 | T4 | 🔴 TELAT_BERAT |
""")
    with col_p:
        st.markdown("""### ⏰ PULANG
| Jam | Kode | Status |
|-----|------|--------|
| ≥ 16:00 | TWP | 🟢 TEPAT_WAKTU_PULANG |
| 15:30–15:59 | PC1 | 🟡 PULANG_CEPAT |
| 15:00–15:29 | PC2 | 🟡 PULANG_CEPAT_RINGAN |
| 14:00–14:59 | PC3 | 🟠 PULANG_CEPAT_SEDANG |
| < 14:00 | PC4 | 🔴 PULANG_CEPAT_BERAT |
""")
    with st.form("pred"):
        c1, c2, c3 = st.columns(3)
        with c1:
            m_kar  = st.number_input("Karyawan ID", min_value=1, value=1001)
            m_skpd = st.selectbox("SKPD",
                sorted(oc['id_skpd'].tolist()) if not oc.empty else [0])
        with c2:
            m_lat  = st.number_input("Latitude",
                value=float(oc['office_lat'].median()) if not oc.empty else -6.2, format="%.6f")
            m_long = st.number_input("Longitude",
                value=float(oc['office_long'].median()) if not oc.empty else 106.8, format="%.6f")
        with c3:
            m_jenis = st.selectbox("Jenis", ['M','P'],
                format_func=lambda x: 'Masuk' if x=='M' else 'Pulang')
            m_waktu = st.time_input("Jam")
        sub = st.form_submit_button("🔮 Prediksi", type="primary", use_container_width=True)
    if sub:
        jam_des = m_waktu.hour + m_waktu.minute / 60.0
        status  = determine_status_from_jam(jam_des, m_jenis)
        berm    = is_bermasalah(status)
        dist_km = 0.0
        if not oc.empty:
            off = oc[oc['id_skpd'] == m_skpd]
            if not off.empty:
                o = off.iloc[0]
                dist_km = haversine(m_lat, m_long, o['office_lat'], o['office_long']) / 1000
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Status", f"{status_emoji(status)} {status}")
        with c2: st.metric("Jarak",  f"{dist_km:.3f} km")
        with c3: st.metric("Jam",    f"{m_waktu.hour:02d}:{m_waktu.minute:02d}")
        if berm: st.error(f"{status_emoji(status)} **{status}** — Perlu perhatian!")
        else:    st.success(f"{status_emoji(status)} **{status}**")
        if dist_km > 0.3:
            st.warning(f"📍 Di luar radius 300m ({dist_km:.3f} km)")

    st.markdown("---")
    pf = st.file_uploader("📂 Batch Prediksi", type=['csv','xlsx'], key='pu')
    if pf and st.button("🚀 Jalankan Batch", type="primary"):
        with st.spinner("⏳ Memproses..."):
            dfp, _, _ = load_processed_file(pf.getvalue(), pf.name)
            if not oc.empty and 'id_skpd' in dfp.columns:
                dfp = dfp.merge(oc, on='id_skpd', how='left')
                dfp['dist_km'] = dfp.apply(
                    lambda r: haversine(r['lat'], r['long'],
                                        r.get('office_lat', r['lat']),
                                        r.get('office_long', r['long'])) / 1000
                    if pd.notna(r.get('office_lat')) else 0.0, axis=1)
        st.success(f"✅ {len(dfp):,} baris selesai diproses")
        vc = dfp['status_presensi'].value_counts()
        nc = min(len(vc), 5)
        cols_m = st.columns(nc)
        for i, (s, c) in enumerate(vc.items()):
            with cols_m[i % nc]: st.metric(f"{status_emoji(s)} {s}", f"{c:,}")
        cols = [c for c in ['karyawan_id','id_skpd','jenis','tanggal_kirim',
                            'status_presensi','dist_km'] if c in dfp.columns]
        st.dataframe(dfp[cols], use_container_width=True)
        st.download_button("⬇️ Download Hasil", dfp[cols].to_csv(index=False).encode(),
                           "prediksi.csv", "text/csv")


# ============================================================
# PREPROCESSING PAGE
# ============================================================

def _run_preprocessing(df_raw: pd.DataFrame, config: dict):
    """
    Pipeline preprocessing lengkap:
    1. Status presensi (T2/T3/T4/TWM/TWP/PC1-PC4) + durasi menit
    2. Time features (jam, menit, jam_desimal, weekday)
    3. Coordinate transform (radian)
    4. DBSCAN spatial clustering
    5. Estimasi centroid kantor
    6. Haversine distance
    7. Validation flags
    8. ST-DBSCAN spatio-temporal
    9. Anomaly score + risk level
    """
    import time
    df = df_raw.copy()
    logs = []

    def log(msg): logs.append(msg)

    # ── STEP 1: Status Presensi & Durasi ─────────────────────
    df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'])

    has_post_dt  = 'masuk_post_dt'  in df.columns
    has_pre_dt   = 'pulang_pre_dt'  in df.columns
    has_post_t   = 'masuk_post_time' in df.columns
    has_pre_t    = 'pulang_pre_time' in df.columns

    # Build datetime threshold dari time columns jika belum ada
    def build_dt(row, tcol):
        t = row[tcol]
        if pd.isna(t): return pd.NaT
        base = row['tanggal_kirim'].normalize()
        return base.replace(hour=t.hour, minute=t.minute, second=t.second)

    if not has_post_dt and has_post_t:
        df['masuk_post_dt'] = df.apply(lambda r: build_dt(r,'masuk_post_time'), axis=1)
    if not has_pre_dt and has_pre_t:
        df['pulang_pre_dt'] = df.apply(lambda r: build_dt(r,'pulang_pre_time'), axis=1)

    mask_m = df['jenis'] == 'M'
    mask_p = df['jenis'] == 'P'

    df['menit_telat']  = np.nan
    df['menit_cepat']  = np.nan

    if 'masuk_post_dt' in df.columns:
        df.loc[mask_m, 'menit_telat'] = (
            (df.loc[mask_m,'tanggal_kirim'] - df.loc[mask_m,'masuk_post_dt'])
            .dt.total_seconds() / 60).round(2)
    if 'pulang_pre_dt' in df.columns:
        df.loc[mask_p, 'menit_cepat'] = (
            (df.loc[mask_p,'pulang_pre_dt'] - df.loc[mask_p,'tanggal_kirim'])
            .dt.total_seconds() / 60).round(2)

    df['durasi_menit'] = df['menit_telat'].fillna(df['menit_cepat'])

    def classify_masuk(m):
        if pd.isna(m): return 'UNKNOWN'
        if m <= 0:  return 'TWM'
        if m <= 14: return 'T2'
        if m <= 44: return 'T3'
        return 'T4'

    def classify_pulang(m):
        if pd.isna(m): return 'UNKNOWN'
        if m <= 0:  return 'TWP'
        if m <= 30: return 'PC1'
        if m <= 60: return 'PC2'
        if m <= 90: return 'PC3'
        return 'PC4'

    if 'status_presensi' not in df.columns or config.get('recalc_status', False):
        df['status_presensi_calc'] = pd.Series(dtype=object)
        df.loc[mask_m, 'status_presensi_calc'] = df.loc[mask_m,'menit_telat'].apply(classify_masuk).values
        df.loc[mask_p, 'status_presensi_calc'] = df.loc[mask_p,'menit_cepat'].apply(classify_pulang).values
        if 'status_presensi' not in df.columns:
            df['status_presensi'] = df['status_presensi_calc']
        n_diff = (df['status_presensi'] != df['status_presensi_calc']).sum()
        log(f"✅ Status presensi dihitung ulang. Mismatch vs asli: {n_diff:,} baris")
    else:
        df['status_presensi_calc'] = pd.Series(dtype=object)
        df.loc[mask_m, 'status_presensi_calc'] = df.loc[mask_m,'menit_telat'].apply(classify_masuk).values
        df.loc[mask_p, 'status_presensi_calc'] = df.loc[mask_p,'menit_cepat'].apply(classify_pulang).values
        log("✅ STEP 1: Durasi telat/cepat dihitung, status_presensi_calc ditambahkan")

    # ── STEP 2: Time Features ─────────────────────────────────
    if 'jam' not in df.columns:
        df['jam']         = df['tanggal_kirim'].dt.hour
        df['menit_waktu'] = df['tanggal_kirim'].dt.minute
        df['jam_desimal'] = df['jam'] + df['menit_waktu'] / 60.0
        df['weekday']     = df['tanggal_kirim'].dt.weekday
        df['tanggal']     = df['tanggal_kirim'].dt.date
        df['timestamp_num'] = df['tanggal_kirim'].astype(np.int64) // 10**9
    log("✅ STEP 2: Time features (jam, jam_desimal, weekday, timestamp_num)")

    # ── STEP 3: Coordinate Transform ─────────────────────────
    df = df.dropna(subset=['lat','long'])
    df['lat_rad']  = np.radians(df['lat'])
    df['long_rad'] = np.radians(df['long'])
    log("✅ STEP 3: Koordinat ke radian (lat_rad, long_rad)")

    if not config.get('run_dbscan', True):
        log("⏭️ STEP 4-5: DBSCAN dilewati (nonaktif)")
        df['cluster_masuk'] = -1; df['cluster_pulang'] = -1
        df['is_noise_masuk'] = 0; df['is_noise_pulang'] = 0
        df['cluster_size_masuk'] = 0; df['cluster_size_pulang'] = 0
    else:
        if not SKLEARN_OK:
            log("⚠️ sklearn tidak tersedia, DBSCAN dilewati")
            df['cluster_masuk'] = -1; df['cluster_pulang'] = -1
            df['is_noise_masuk'] = 0; df['is_noise_pulang'] = 0
            df['cluster_size_masuk'] = 0; df['cluster_size_pulang'] = 0
        else:
            # ── STEP 4: DBSCAN ───────────────────────────────
            eps_km  = config.get('eps_km', 0.1)
            min_smp = config.get('min_samples', 3)
            eps_rad = eps_km / 6371.0

            def dbscan_spatial(subset, jenis_col):
                result = pd.Series(-1, index=df.index, dtype=int)
                for skpd_id, grp in subset.groupby('id_skpd'):
                    if len(grp) < min_smp: continue
                    coords = grp[['lat_rad','long_rad']].values
                    labels = DBSCAN(eps=eps_rad, min_samples=min_smp,
                                    algorithm='ball_tree', metric='haversine').fit(coords).labels_
                    max_l = result.max() + 1
                    result.loc[grp.index] = np.where(labels>=0, labels+max_l, -1)
                return result

            df['cluster_masuk']  = dbscan_spatial(df[df['jenis']=='M'], 'M')
            df['cluster_pulang'] = dbscan_spatial(df[df['jenis']=='P'], 'P')
            df['is_noise_masuk']  = (df['cluster_masuk']  == -1).astype(int)
            df['is_noise_pulang'] = (df['cluster_pulang'] == -1).astype(int)
            for jenis, col in [('M','cluster_masuk'),('P','cluster_pulang')]:
                sz_col = col.replace('cluster','cluster_size')
                mask = df['jenis']==jenis
                sz = df[mask].groupby(col)['karyawan_id'].transform('count')
                df.loc[mask, sz_col] = sz
            df[['cluster_size_masuk','cluster_size_pulang']] = (
                df[['cluster_size_masuk','cluster_size_pulang']].fillna(0).astype(int))
            n_cluster_m = (df['cluster_masuk'] >= 0).sum()
            n_cluster_p = (df['cluster_pulang'] >= 0).sum()
            log(f"✅ STEP 4: DBSCAN — {n_cluster_m:,} masuk / {n_cluster_p:,} pulang di-cluster")

            # ── STEP 5: Centroid Kantor ───────────────────────
            office_locs = {}
            for skpd_id, grp in df[df['jenis']=='M'].groupby('id_skpd'):
                non_noise = grp[grp['cluster_masuk'] != -1]
                if len(non_noise) == 0:
                    office_locs[skpd_id] = {'office_lat':grp['lat'].median(),'office_long':grp['long'].median()}
                else:
                    best = non_noise['cluster_masuk'].value_counts().idxmax()
                    pts  = non_noise[non_noise['cluster_masuk']==best]
                    office_locs[skpd_id] = {'office_lat':pts['lat'].mean(),'office_long':pts['long'].mean()}
            oc_df = pd.DataFrame.from_dict(office_locs, orient='index').reset_index()
            oc_df.columns = ['id_skpd','office_lat','office_long']
            oc_df['id_skpd'] = oc_df['id_skpd'].astype(df['id_skpd'].dtype)
            if 'office_lat' in df.columns: df = df.drop(columns=['office_lat','office_long'])
            df = df.merge(oc_df, on='id_skpd', how='left')
            log(f"✅ STEP 5: Centroid kantor untuk {len(office_locs)} SKPD")

    # ── STEP 6: Haversine Distance ────────────────────────────
    if 'office_lat' not in df.columns:
        # Fallback: hitung centroid sederhana jika DBSCAN dilewati
        oc_simple = df[df['jenis']=='M'].groupby('id_skpd')[['lat','long']].median().reset_index()
        oc_simple.columns = ['id_skpd','office_lat','office_long']
        df = df.merge(oc_simple, on='id_skpd', how='left')
        log("✅ STEP 5b: Centroid kantor dari median (tanpa DBSCAN)")

    def haversine_vec(lat1,lon1,lat2,lon2):
        R = 6371.0
        r1,r2,r3,r4 = map(np.radians,[lat1,lon1,lat2,lon2])
        a = np.sin((r2-r1)/2)**2 + np.cos(r1)*np.cos(r2)*np.sin((r4-r3)/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    df['dist_km']      = haversine_vec(df['lat'].values,df['long'].values,df['office_lat'].values,df['office_long'].values)
    df['outside_300m'] = (df['dist_km'] > 0.3).astype(int)
    df['very_far']     = (df['dist_km'] > 5.0).astype(int)
    df['extreme_far']  = (df['dist_km'] > 50.0).astype(int)
    log(f"✅ STEP 6: Haversine — {df['outside_300m'].sum():,} di luar 300m, {df['very_far'].sum():,} >5km")

    # ── STEP 7: Validation Flags ──────────────────────────────
    df['no_note']          = df['catatan'].isna().astype(int)
    df['far_no_note']      = ((df['outside_300m']==1)&(df['no_note']==1)).astype(int)
    df['far_with_note']    = ((df['outside_300m']==1)&(df['no_note']==0)).astype(int)
    df['near_but_status0'] = ((df['outside_300m']==0)&(df['status_lokasi']==0)).astype(int)
    df['far_but_status1']  = ((df['outside_300m']==1)&(df['status_lokasi']==1)).astype(int)
    log(f"✅ STEP 7: Validation — far_no_note:{df['far_no_note'].sum():,}, far_with_note:{df['far_with_note'].sum():,}")

    # ── STEP 8: ST-DBSCAN ─────────────────────────────────────
    if not config.get('run_stdbscan', True) or not SKLEARN_OK:
        df['st_cluster_masuk']  = -1; df['st_cluster_pulang']  = -1
        df['is_st_noise_masuk'] = 0;  df['is_st_noise_pulang'] = 0
        log("⏭️ STEP 8: ST-DBSCAN dilewati")
    else:
        st_eps_km   = config.get('st_eps_km', 0.1)
        st_eps_hr   = config.get('st_eps_hours', 1.0)
        st_min_smp  = config.get('st_min_samples', 3)
        eps_deg     = st_eps_km / 111.0
        time_scale  = eps_deg / st_eps_hr

        def stdbscan(subset):
            result = pd.Series(-1, index=df.index, dtype=int)
            for skpd_id, grp in subset.groupby('id_skpd'):
                if len(grp) < st_min_smp: continue
                X = grp[['lat','long']].copy()
                X['jam_sc'] = grp['jam_desimal'] * time_scale
                labels = DBSCAN(eps=eps_deg, min_samples=st_min_smp,
                                algorithm='auto', metric='euclidean').fit(X.values).labels_
                max_l = result.max() + 1
                result.loc[grp.index] = np.where(labels>=0, labels+max_l, -1)
            return result

        df['st_cluster_masuk']  = stdbscan(df[df['jenis']=='M'])
        df['st_cluster_pulang'] = stdbscan(df[df['jenis']=='P'])
        df['is_st_noise_masuk']  = (df['st_cluster_masuk']  == -1).astype(int)
        df['is_st_noise_pulang'] = (df['st_cluster_pulang'] == -1).astype(int)
        log(f"✅ STEP 8: ST-DBSCAN — noise masuk:{df['is_st_noise_masuk'].sum():,} pulang:{df['is_st_noise_pulang'].sum():,}")

    # ── STEP 9: Anomaly Score & Risk Level ────────────────────
    sc = pd.Series(0, index=df.index)
    sc += df.get('extreme_far',       pd.Series(0,index=df.index)) * 3
    sc += df.get('very_far',          pd.Series(0,index=df.index)) * 2
    sc += df.get('far_no_note',       pd.Series(0,index=df.index)) * 2
    sc += df.get('outside_300m',      pd.Series(0,index=df.index)) * 1
    sc += df.get('is_noise_masuk',    pd.Series(0,index=df.index)) * 1
    sc += df.get('is_st_noise_masuk', pd.Series(0,index=df.index)) * 1
    sc += df.get('far_but_status1',   pd.Series(0,index=df.index)) * 1
    sc -= df.get('far_with_note',     pd.Series(0,index=df.index)) * 1
    sc  = sc.clip(lower=0)
    df['anomaly_score'] = sc.astype(int)
    df['risk_level'] = pd.cut(df['anomaly_score'], bins=[-1,1,3,99],
                               labels=['LOW','MED','HIGH'])
    rl = df['risk_level'].value_counts()
    log(f"✅ STEP 9: Anomaly score — HIGH:{rl.get('HIGH',0):,} MED:{rl.get('MED',0):,} LOW:{rl.get('LOW',0):,}")


    # ── STEP 10: Pilih & urutkan kolom output sesuai format target ──────────
    # Kolom ekstra dari preprocessing (tetap disimpan sebagai referensi)
    # status_presensi_calc dan menit_telat/cepat tetap ada untuk validasi
    keep_cols = [
        # Core
        'karyawan_id','lat','long','tanggal_kirim','jenis',
        'id_skpd','status_lokasi','catatan','approver_status',
        'status_presensi',
        # Time features
        'timestamp_num','jam','menit','jam_desimal','tanggal','weekday',
        # Spatial
        'lat_rad','long_rad','office_lat','office_long','dist_km',
        # Distance flags
        'outside_300m','very_far','extreme_far',
        # Validation flags
        'no_note','far_no_note','far_with_note','near_but_status0','far_but_status1',
        # DBSCAN
        'cluster_masuk','cluster_pulang',
        'is_noise_masuk','is_noise_pulang',
        'cluster_size_masuk','cluster_size_pulang',
        # ST-DBSCAN
        'st_cluster_masuk','st_cluster_pulang',
        'is_st_noise_masuk','is_st_noise_pulang',
        # Anomaly
        'anomaly_score','risk_level',
        # Extra dari preprocessing baru (untuk validasi)
        'menit_telat','menit_cepat','durasi_menit','status_presensi_calc',
    ]
    # Tambahkan kolom raw yang ada tapi belum di keep_cols
    existing_extra = [c for c in df.columns if c not in keep_cols]
    final_cols = [c for c in keep_cols if c in df.columns] + existing_extra
    df = df[final_cols]

    log(f"✅ STEP 10: Output {len(df.columns)} kolom, {len(df):,} baris")
    return df, logs


def page_preprocessing():
    st.markdown("## 🔧 Preprocessing Data Mentah")
    st.markdown("Upload file absensi **mentah** → otomatis diproses → download hasil.")

    uploaded_raw = st.file_uploader(
        "📂 Upload file mentah (CSV / Excel)",
        type=['csv','xlsx'], key='pp_upload')

    if uploaded_raw is None:
        st.info("💡 Upload file absensi mentah untuk mulai preprocessing.")
        with st.expander("ℹ️ Kolom yang dibutuhkan", expanded=False):
            st.markdown("""
| Kolom | Keterangan |
|---|---|
| `karyawan_id` | ID pegawai |
| `id_skpd` | ID kantor/SKPD |
| `jenis` | M (masuk) / P (pulang) |
| `lat`, `long` | Koordinat GPS |
| `tanggal_kirim` | Waktu absensi |
| `status_lokasi` | 0/1 dari sistem lama |
| `catatan` | Keterangan alasan |
| `masuk_post_time` **atau** `masuk_post_dt` | Batas toleransi masuk |
| `pulang_pre_time` **atau** `pulang_pre_dt` | Jam minimum pulang |
| `status_presensi` | Opsional (untuk validasi) |
""")
        return

    # ── Konfigurasi ────────────────────────────────────────────
    st.markdown("### ⚙️ Konfigurasi Pipeline")
    c1, c2, c3 = st.columns(3)
    with c1:
        run_dbscan   = st.checkbox("🔵 Jalankan DBSCAN",    value=True,
                                   help="Clustering spasial per SKPD")
        run_stdbscan = st.checkbox("🟣 Jalankan ST-DBSCAN", value=True,
                                   help="Clustering spasial+temporal")
        recalc       = st.checkbox("🔄 Hitung ulang status_presensi", value=False,
                                   help="Override kolom status_presensi yang sudah ada")
    with c2:
        eps_km     = st.number_input("DBSCAN radius (km)",    value=0.1, step=0.05, format="%.2f")
        min_smp    = st.number_input("DBSCAN min_samples",    value=3, step=1)
        st_eps_km  = st.number_input("ST-DBSCAN radius (km)", value=0.1, step=0.05, format="%.2f")
    with c3:
        st_eps_hr  = st.number_input("ST-DBSCAN radius (jam)", value=1.0, step=0.5, format="%.1f")
        st_min_smp = st.number_input("ST-DBSCAN min_samples",  value=3, step=1)

    config = {
        'run_dbscan':    run_dbscan,
        'run_stdbscan':  run_stdbscan,
        'recalc_status': recalc,
        'eps_km':        eps_km,
        'min_samples':   int(min_smp),
        'st_eps_km':     st_eps_km,
        'st_eps_hours':  st_eps_hr,
        'st_min_samples':int(st_min_smp),
    }

    if not st.button("🚀 Jalankan Preprocessing", type="primary", use_container_width=True):
        return

    # ── Load raw data ──────────────────────────────────────────
    with st.spinner("⏳ Membaca file..."):
        buf = io.BytesIO(uploaded_raw.getvalue())
        try:
            if uploaded_raw.name.endswith('.csv'):
                df_raw = pd.read_csv(buf)
            else:
                df_raw = pd.read_excel(buf)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}"); return

    st.info(f"📊 File dibaca: **{len(df_raw):,} baris**, **{len(df_raw.columns)} kolom**")

    # ── Cek kolom wajib ────────────────────────────────────────
    required = ['karyawan_id','id_skpd','jenis','lat','long','tanggal_kirim']
    missing  = [c for c in required if c not in df_raw.columns]
    if missing:
        st.error(f"❌ Kolom wajib tidak ada: `{missing}`"); return

    # ── Jalankan pipeline ──────────────────────────────────────
    progress = st.progress(0, text="Memulai preprocessing...")
    try:
        progress.progress(10, "Step 1-2: Status & Time features...")
        df_out, logs = _run_preprocessing(df_raw, config)
        progress.progress(100, "✅ Selesai!")
    except Exception as e:
        import traceback
        st.error(f"❌ Error: {e}")
        st.code(traceback.format_exc())
        return

    # ── Tampilkan log ──────────────────────────────────────────
    st.markdown("### 📋 Log Preprocessing")
    for log in logs:
        st.markdown(f"- {log}")

    # ── Ringkasan hasil ────────────────────────────────────────
    st.markdown("### 📊 Ringkasan Hasil")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Baris",  f"{len(df_out):,}")
    with col2: st.metric("Karyawan",     f"{df_out['karyawan_id'].nunique():,}")
    with col3: st.metric("SKPD",         f"{df_out['id_skpd'].nunique():,}")
    with col4:
        n_high = (df_out.get('risk_level', pd.Series()) == 'HIGH').sum()
        st.metric("🔴 HIGH Risk", f"{n_high:,}")

    # Status distribution
    if 'status_presensi_calc' in df_out.columns:
        st.markdown("#### Status Presensi (hasil kalkulasi)")
        vc = df_out['status_presensi_calc'].value_counts()
        order = ['TWM','T2','T3','T4','TWP','PC1','PC2','PC3','PC4']
        cols_s = st.columns(min(len(vc), 5))
        for i, s in enumerate([x for x in order if x in vc.index]):
            with cols_s[i % 5]:
                st.metric(f"{status_emoji(s)} {s}", f"{vc[s]:,}")

    if 'status_presensi' in df_raw.columns and 'status_presensi_calc' in df_out.columns:
        n_match = (df_out['status_presensi'] == df_out['status_presensi_calc']).sum()
        pct = n_match / len(df_out) * 100
        if pct >= 99:
            st.success(f"✅ Validasi status: **{n_match:,}/{len(df_out):,}** match ({pct:.2f}%)")
        else:
            st.warning(f"⚠️ Validasi status: **{n_match:,}/{len(df_out):,}** match ({pct:.2f}%) — ada perbedaan")

    # Risk level
    if 'risk_level' in df_out.columns:
        st.markdown("#### Risk Level")
        rc1, rc2, rc3 = st.columns(3)
        rl = df_out['risk_level'].value_counts()
        with rc1: st.metric("🔴 HIGH", f"{rl.get('HIGH',0):,}")
        with rc2: st.metric("🟠 MED",  f"{rl.get('MED',0):,}")
        with rc3: st.metric("🟢 LOW",  f"{rl.get('LOW',0):,}")

    # Kolom baru
    new_cols = ['menit_telat','menit_cepat','durasi_menit','status_presensi_calc',
                'dist_km','outside_300m','very_far','extreme_far',
                'no_note','far_no_note','far_with_note',
                'cluster_masuk','is_noise_masuk','is_st_noise_masuk',
                'anomaly_score','risk_level']
    added = [c for c in new_cols if c in df_out.columns and c not in df_raw.columns]
    if added:
        st.markdown(f"#### ✨ Kolom baru ditambahkan ({len(added)})")
        st.code(", ".join(added))

    # Preview
    with st.expander("🔍 Preview 10 baris pertama"):
        preview_cols = ['karyawan_id','jenis','tanggal_kirim','status_presensi',
                        'status_presensi_calc','menit_telat','menit_cepat',
                        'dist_km','anomaly_score','risk_level']
        preview_cols = [c for c in preview_cols if c in df_out.columns]
        st.dataframe(df_out[preview_cols].head(10), use_container_width=True)

    # ── Download & Gunakan ────────────────────────────────────
    st.markdown("### 💾 Download & Gunakan Data")
    dl1, dl2 = st.columns(2)

    with dl1:
        st.markdown("**⬇️ Download saja**")
        dc1, dc2 = st.columns(2)
        with dc1:
            csv_buf = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("📄 CSV",
                               csv_buf, "absensi_preprocessed.csv", "text/csv",
                               use_container_width=True)
        with dc2:
            xl_buf = io.BytesIO()
            df_out.to_excel(xl_buf, index=False)
            st.download_button("📊 Excel",
                               xl_buf.getvalue(),
                               "absensi_preprocessed.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

    with dl2:
        st.markdown("**▶️ Simpan & Gunakan langsung**")
        save_name = st.text_input("Nama file", value="absensi_preprocessed.csv",
                                  key="pp_save_name",
                                  help="File akan disimpan di direktori app dan muncul di Pilih file lokal")
        if st.button("💾 Simpan & Load ke App", type="primary", use_container_width=True):
            try:
                # Simpan ke disk agar muncul di scan_local_files()
                save_path = save_name if save_name.endswith(('.csv','.xlsx')) else save_name + '.csv'
                if save_path.endswith('.xlsx'):
                    xl_buf2 = io.BytesIO()
                    df_out.to_excel(xl_buf2, index=False)
                    with open(save_path, 'wb') as f_:
                        f_.write(xl_buf2.getvalue())
                else:
                    df_out.to_csv(save_path, index=False)

                # Langsung load ke session
                df_mapped, fc, remaps = load_processed_file(
                    df_out.to_csv(index=False).encode(), save_path)
                st.session_state['df'] = df_mapped
                st.session_state['office_centroid'] = build_office_centroid(df_mapped)
                st.session_state['file_name'] = save_path
                # Clear cache agar scan_local_files() menemukan file baru
                st.cache_data.clear()
                st.success(f"✅ Disimpan sebagai **{save_path}** dan sudah di-load!")
                st.info("Sekarang bisa ke halaman Visualisasi, atau pilih file ini di Upload Data → Pilih file lokal.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal menyimpan: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Clear stale cache jika versi berubah (return signature load_processed_file = 3 nilai)
    if not st.session_state.get('_cache_cleared_v3'):
        st.cache_data.clear()
        st.session_state['_cache_cleared_v3'] = True

    for key, default in [('df', None), ('office_centroid', pd.DataFrame()),
                         ('watchlist', [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state['df'] is None and not st.session_state.get('_al'):
        st.session_state['_al'] = True
        lf = scan_local_files()
        if lf:
            try:
                df, _, _ = load_local_file(lf[0])
                st.session_state['df'] = df
                st.session_state['office_centroid'] = build_office_centroid(df)
                st.session_state['file_name'] = lf[0]
                st.session_state['_autoloaded'] = lf[0]
            except Exception:
                pass

    page, uploaded, filters = render_sidebar()

    if uploaded is not None:
        fh = hashlib.md5(uploaded.getvalue()).hexdigest()
        if st.session_state.get('_fh') != fh:
            df, _, _ = load_processed_file(uploaded.getvalue(), uploaded.name)
            st.session_state['df'] = df
            st.session_state['office_centroid'] = build_office_centroid(df)
            st.session_state['_fh'] = fh
            st.session_state['file_name'] = uploaded.name
            st.session_state.pop('_autoloaded', None)

    if st.session_state.get('_autoloaded') and st.session_state['df'] is not None:
        st.sidebar.success(
            f"✅ Auto-load: **{st.session_state['_autoloaded']}**\n"
            f"{len(st.session_state['df']):,} baris")

    pages = {
        "🏠 Beranda":        page_beranda,
        "📥 Upload Data":    lambda: page_upload(uploaded),
        "🔧 Preprocessing":  page_preprocessing,
        "📊 Visualisasi":    lambda: page_visualisasi(filters),
        "🎯 Hunting":        page_hunting,
        "🔮 Prediksi":       page_prediksi,
    }
    pages.get(page, page_beranda)()

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:gray;font-size:11px">'
        'Analisis Absensi v3 — T2/T3/T4/TWM/TWP/PC1-4 mapping | Streamlit + Folium + Plotly'
        '</p>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
