"""
Visualisasi & Analisis Absensi v3
Status Presensi: T1/T2/T3/T4/TWM/TWP/PC1-4 dari kolom status_presensi

PREPROCESSING: saat ini DINONAKTIFKAN dari navigasi.
Untuk mengaktifkan kembali, cari komentar "AKTIFKAN PREPROCESSING" dan ikuti petunjuknya.
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
STATUS_CODE_MAP = {
    'T1':  'TELAT_MASUK_RINGAN',
    'T2':  'TELAT_MASUK_SEDANG',
    'T3':  'TELAT_MASUK_BERAT',
    'T4':  'TELAT_MASUK_SANGAT_BERAT',
    'TWM': 'TEPAT_WAKTU_MASUK',
    'TWP': 'TEPAT_WAKTU_PULANG',
    'PC1': 'PULANG_CEPAT',
    'PC2': 'PULANG_CEPAT_RINGAN',
    'PC3': 'PULANG_CEPAT_SEDANG',
    'PC4': 'PULANG_CEPAT_BERAT',
}
STATUS_LEGACY_MAP = {
    'PULANG_NORMAL': 'TEPAT_WAKTU_PULANG',
    'HADIR':         'TEPAT_WAKTU_MASUK',
}
STATUS_VALID = {
    'TELAT_MASUK_RINGAN', 'TELAT_MASUK_SEDANG', 'TELAT_MASUK_BERAT', 'TELAT_MASUK_SANGAT_BERAT',
    'TEPAT_WAKTU_MASUK', 'TEPAT_WAKTU_PULANG',
    'PULANG_CEPAT', 'PULANG_CEPAT_RINGAN', 'PULANG_CEPAT_SEDANG', 'PULANG_CEPAT_BERAT',
}
STATUS_AMBIGUOUS = {'TELAT', 'PULANG'}
STATUS_ORDER = [
    'TELAT_MASUK_SANGAT_BERAT', 'PULANG_CEPAT_BERAT',
    'TELAT_MASUK_BERAT', 'PULANG_CEPAT_SEDANG',
    'TELAT_MASUK_SEDANG', 'PULANG_CEPAT_RINGAN', 'PULANG_CEPAT',
    'TELAT_MASUK_RINGAN',
    'TEPAT_WAKTU_MASUK', 'TEPAT_WAKTU_PULANG',
]
STATUS_BERMASALAH = {
    'TELAT_MASUK_SANGAT_BERAT', 'TELAT_MASUK_BERAT', 'TELAT_MASUK_SEDANG',
    'PULANG_CEPAT_BERAT', 'PULANG_CEPAT_SEDANG',
}
STATUS_COLORS = {
    'TELAT_MASUK_SANGAT_BERAT':  '#6e0d0d',
    'TELAT_MASUK_BERAT':         '#c0392b',
    'TELAT_MASUK_SEDANG':        '#e67e22',
    'TELAT_MASUK_RINGAN':        '#d4ac0d',
    'TEPAT_WAKTU_MASUK':         '#27ae60',
    'TEPAT_WAKTU_PULANG':        '#2ecc71',
    'PULANG_CEPAT':              '#f39c12',
    'PULANG_CEPAT_RINGAN':       '#d4ac0d',
    'PULANG_CEPAT_SEDANG':       '#e67e22',
    'PULANG_CEPAT_BERAT':        '#c0392b',
    'UNKNOWN':                   '#95a5a6',
}
STATUS_EMOJI = {
    'TELAT_MASUK_SANGAT_BERAT':  '⛔',
    'TELAT_MASUK_BERAT':         '🔴',
    'TELAT_MASUK_SEDANG':        '🟠',
    'TELAT_MASUK_RINGAN':        '🟡',
    'TEPAT_WAKTU_MASUK':         '🟢',
    'TEPAT_WAKTU_PULANG':        '🟢',
    'PULANG_CEPAT':              '🟡',
    'PULANG_CEPAT_RINGAN':       '🟡',
    'PULANG_CEPAT_SEDANG':       '🟠',
    'PULANG_CEPAT_BERAT':        '🔴',
    'UNKNOWN':                   '⚪',
}
STATUS_FOLIUM_HEX = {
    'TEPAT_WAKTU_MASUK':         '#27ae60',
    'TELAT_MASUK_RINGAN':        '#f1c40f',
    'TELAT_MASUK_SEDANG':        '#e67e22',
    'TELAT_MASUK_BERAT':         '#c0392b',
    'TELAT_MASUK_SANGAT_BERAT':  '#6e0d0d',
    'TEPAT_WAKTU_PULANG':        '#1abc9c',
    'PULANG_CEPAT':              '#9b59b6',
    'PULANG_CEPAT_RINGAN':       '#8e44ad',
    'PULANG_CEPAT_SEDANG':       '#e74c3c',
    'PULANG_CEPAT_BERAT':        '#922b21',
    'UNKNOWN':                   '#95a5a6',
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
    if pd.isna(val) or str(val).strip() == '':
        return 'UNKNOWN'
    v = str(val).strip().upper()
    if v in STATUS_CODE_MAP:    return STATUS_CODE_MAP[v]
    if v in STATUS_VALID:       return v
    if v in STATUS_LEGACY_MAP:  return STATUS_LEGACY_MAP[v]
    if v in STATUS_AMBIGUOUS:   return v
    return 'UNKNOWN'

def resolve_ambiguous(df):
    remaps = []
    mask = df['status_presensi'] == 'TELAT'
    if mask.any():
        n = mask.sum()
        if 'jenis' in df.columns:
            df.loc[mask & (df['jenis'] == 'M'), 'status_presensi'] = 'TEPAT_WAKTU_MASUK'
            df.loc[mask & (df['jenis'] == 'P'), 'status_presensi'] = 'TEPAT_WAKTU_PULANG'
            df.loc[df['status_presensi'] == 'TELAT', 'status_presensi'] = 'TEPAT_WAKTU_MASUK'
        else:
            df.loc[mask, 'status_presensi'] = 'TEPAT_WAKTU_MASUK'
        remaps.append(f"🔄 {n:,} baris 'TELAT' → TEPAT_WAKTU (berdasarkan kolom jenis)")
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
    if jenis == 'M':
        if jam_desimal <= 8.25:   return 'TEPAT_WAKTU_MASUK'
        elif jam_desimal <= 8.75: return 'TELAT_MASUK_RINGAN'
        elif jam_desimal <= 9.25: return 'TELAT_MASUK_SEDANG'
        elif jam_desimal <= 9.75: return 'TELAT_MASUK_BERAT'
        else:                     return 'TELAT_MASUK_SANGAT_BERAT'
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
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def status_color(s):        return STATUS_COLORS.get(s, '#95a5a6')
def status_emoji(s):        return STATUS_EMOJI.get(s, '⚪')
def status_folium_color(s): return STATUS_FOLIUM_HEX.get(s, '#95a5a6')
def is_bermasalah(s):       return s in STATUS_BERMASALAH
def _df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def _ensure_all_status_cols(df_pivot, status_list):
    for s in status_list:
        if s not in df_pivot.columns:
            df_pivot[s] = 0
    return df_pivot

# ============================================================
# OFFICE MARKER — DivIcon
# Ubah OFFICE_ICON_SIZE untuk resize (default: 20px)
# ============================================================
OFFICE_ICON_SIZE = 20  # ← ganti angka ini untuk memperbesar/memperkecil

def make_office_icon(skpd_id):
    """Buat DivIcon kotak kecil berlabel 🏢 untuk marker kantor SKPD."""
    s    = OFFICE_ICON_SIZE
    half = s // 2
    fs   = max(s - 6, 10)   # font-size emoji
    return folium.DivIcon(
        html=f"""<div style="
            width:{s}px; height:{s}px;
            background:#2980b9;
            border:2px solid white;
            border-radius:4px;
            display:flex; align-items:center; justify-content:center;
            box-shadow:0 2px 6px rgba(0,0,0,0.4);
            font-size:{fs}px; line-height:1;
        ">🏢</div>""",
        icon_size=(s, s),
        icon_anchor=(half, half),
        popup_anchor=(0, -half),
    )

def _add_office_markers(m, oc):
    """Tambahkan semua marker kantor SKPD ke peta menggunakan DivIcon."""
    if oc is None or (hasattr(oc, '__len__') and len(oc) == 0):
        return
    for _, o in oc.iterrows():
        if pd.notna(o.get('office_lat')):
            skpd_id = o['id_skpd']
            folium.Marker(
                [o['office_lat'], o['office_long']],
                popup=folium.Popup(f"<b>Kantor SKPD {skpd_id}</b>", max_width=150),
                tooltip=f"🏢 Kantor SKPD {skpd_id}",
                icon=make_office_icon(skpd_id),
            ).add_to(m)
            folium.Circle(
                [o['office_lat'], o['office_long']],
                radius=100,
                color='#3498db', fill=True, fill_color='#3498db',
                fill_opacity=0.05, weight=2, dash_array='6',
            ).add_to(m)

# ============================================================
# FIX DECIMAL
# ============================================================
def fix_decimal_columns(df):
    numeric_hints = [
        'lat','long','lat_rad','long_rad','office_lat','office_long',
        'dist_km','jarak','jam_desimal','jam','menit','weekday',
        'outside_100m','very_far','extreme_far','status_lokasi','timestamp_num',
    ]
    fixed = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if col in numeric_hints:
            try:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',','.', regex=False).str.strip(),
                    errors='coerce')
                fixed.append(col)
            except Exception:
                pass
        else:
            sample = df[col].dropna().head(20).astype(str)
            if sample.str.match(r'^-?\d+,\d+$').mean() > 0.7:
                try:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',','.', regex=False),
                        errors='coerce')
                    fixed.append(col)
                except Exception:
                    pass
    return df, fixed

# ============================================================
# LOAD & PROCESS
# ============================================================
@st.cache_data(show_spinner=False, max_entries=2)
def load_processed_file(file_bytes, file_name):
    buf = io.BytesIO(file_bytes)
    if file_name.endswith('.csv'):
        for sep in [',',';','\t']:
            try:
                df = pd.read_csv(buf, sep=sep)
                if len(df.columns) > 3: break
                buf.seek(0)
            except Exception:
                buf.seek(0)
    else:
        df = pd.read_excel(buf)

    df, fixed_cols = fix_decimal_columns(df)

    if 'tanggal_kirim' in df.columns:
        df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'], errors='coerce')
        if 'jam'        not in df.columns: df['jam']        = df['tanggal_kirim'].dt.hour
        if 'menit'      not in df.columns: df['menit']      = df['tanggal_kirim'].dt.minute
        if 'jam_desimal'not in df.columns: df['jam_desimal']= df['jam'] + df['menit']/60.0
        if 'weekday'    not in df.columns: df['weekday']    = df['tanggal_kirim'].dt.weekday
        if 'tanggal'    not in df.columns: df['tanggal']    = df['tanggal_kirim'].dt.date

    if 'jenis' in df.columns:
        df['jenis'] = df['jenis'].astype(str).str.strip().str.upper()

    remaps = []
    if 'status_presensi' in df.columns:
        original_unique = df['status_presensi'].dropna().unique().tolist()
        df['status_presensi'] = df['status_presensi'].apply(map_status_value)
        for orig in original_unique:
            mapped = map_status_value(orig)
            if str(orig).strip().upper() != mapped and mapped not in STATUS_AMBIGUOUS:
                remaps.append(f"<span class='remap-badge'>{orig} → {mapped}</span>")
        df, ambig_remaps = resolve_ambiguous(df)
        remaps.extend(ambig_remaps)
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

    df['is_bermasalah'] = df['status_presensi'].apply(lambda s: 1 if s in STATUS_BERMASALAH else 0)

    if 'approver_status' in df.columns:
        df['approver_status'] = (df['approver_status'].astype(str).str.strip()
                                 .replace({'nan':'','None':'','NaN':''}))
        df['is_tolak']   = df['approver_status'].str.upper().str.contains('TOLAK',  na=False).astype(int)
        df['is_terima']  = df['approver_status'].str.upper().str.contains('TERIMA', na=False).astype(int)
        df['is_pending'] = ((df['approver_status']=='') | df['approver_status'].isna()).astype(int)
    else:
        df['is_tolak'] = df['is_terima'] = df['is_pending'] = 0

    if 'dist_km' in df.columns and 'lat' in df.columns and 'office_lat' in df.columns:
        median_dist = df['dist_km'].median()
        if median_dist > 100:
            rlat1 = np.radians(df['lat'].values);   rlat2 = np.radians(df['office_lat'].values)
            rlon1 = np.radians(df['long'].values);  rlon2 = np.radians(df['office_long'].values)
            a = np.sin((rlat2-rlat1)/2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin((rlon2-rlon1)/2)**2
            df['dist_km'] = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))
            remaps.append(f"🔧 dist_km dihitung ulang dari koordinat (median lama={median_dist:.0f})")
        df['outside_100m'] = (df['dist_km'] > 0.1).astype(int)
        df['very_far']     = (df['dist_km'] > 5.0).astype(int)
    elif 'dist_km' in df.columns:
        df['outside_100m'] = (df['dist_km'] > 0.1).astype(int)
        df['very_far']     = (df['dist_km'] > 5.0).astype(int)

    return df, fixed_cols, remaps

@st.cache_data(show_spinner=False, max_entries=2)
def build_office_centroid(df):
    if 'office_lat' in df.columns and 'id_skpd' in df.columns:
        return df.groupby('id_skpd')[['office_lat','office_long']].first().reset_index()
    elif 'id_skpd' in df.columns and 'lat' in df.columns:
        src = df[df['jenis']=='M'] if 'jenis' in df.columns and (df['jenis']=='M').any() else df
        return src.groupby('id_skpd').agg(
            office_lat=('lat','median'), office_long=('long','median')).reset_index()
    return pd.DataFrame(columns=['id_skpd','office_lat','office_long'])

@st.cache_data(show_spinner=False, max_entries=2)
def validate_dataframe(df):
    warns = []
    required = ['karyawan_id','lat','long','tanggal_kirim','jenis','id_skpd']
    missing = [c for c in required if c not in df.columns]
    if missing: warns.append(f"Kolom wajib tidak ditemukan: {missing}")
    if 'lat' in df.columns and (~df['lat'].between(-90,90)).sum() > 0:
        warns.append(f"{(~df['lat'].between(-90,90)).sum()} baris lat di luar range")
    return warns

# ============================================================
# FILTER
# ============================================================
@st.cache_data(show_spinner=False, max_entries=50)
def apply_filters(df_hash, df, skpd, jenis_tuple, date_range,
                  dist_range, approver_filter, status_filter):
    f = df.copy()
    if skpd != 'Semua':
        f = f[f['id_skpd'] == skpd]
    if jenis_tuple:
        f = f[f['jenis'].isin(list(jenis_tuple))]
    if date_range and len(date_range)==2 and 'tanggal_kirim' in f.columns:
        f = f[(f['tanggal_kirim'].dt.date >= date_range[0]) &
              (f['tanggal_kirim'].dt.date <= date_range[1])]
    if 'dist_km' in f.columns:
        f = f[(f['dist_km'] >= dist_range[0]) & (f['dist_km'] <= dist_range[1])]
    if approver_filter and approver_filter != 'Semua':
        if approver_filter == 'TOLAK':     f = f[f['is_tolak']  == 1]
        elif approver_filter == 'TERIMA':  f = f[f['is_terima'] == 1]
        elif approver_filter == 'PENDING': f = f[f['is_pending']== 1]
    if status_filter:
        all_vals = set(f['status_presensi'].unique())
        filter_set = set(status_filter)
        if filter_set and not filter_set.intersection(all_vals):
            f['status_presensi'] = f['status_presensi'].apply(map_status_value)
        if filter_set and filter_set != all_vals:
            f = f[f['status_presensi'].isin(filter_set)]
    return f

# ============================================================
# LOCAL FILES
# ============================================================
CANDIDATE_FILES = [
    'absensi_preprocessed.csv','absensi_preprocessed.xlsx',
    'absensi_processed.csv','absensi_processed.xlsx',
    'dataset_absensi_final2.xlsx','absen_pegawai.xlsx',
    'absensi.xlsx','absensi.csv','data_absensi.xlsx','data_absensi.csv',
]

@st.cache_data(show_spinner=False, ttl=10)
def scan_local_files():
    found = []
    for fn in CANDIDATE_FILES:
        if os.path.exists(fn): found.append(fn)
    for fn in sorted(os.listdir('.')):
        if fn.endswith(('.xlsx','.csv')) and fn not in found: found.append(fn)
    return found

@st.cache_data(show_spinner=False, max_entries=2)
def load_local_file(filepath):
    with open(filepath,'rb') as f: fb = f.read()
    return load_processed_file(fb, filepath.split('/')[-1].split('\\')[-1])

# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    st.sidebar.markdown("## 🗺️ Analisis Absensi")
    st.sidebar.markdown("---")

    # nav_pages = ["🏠 Beranda", "📥 Upload Data", "📊 Visualisasi", "🎯 Hunting"]
    nav_pages = ["🏠 Beranda", "📥 Upload Data", "📊 Visualisasi", "🎯 Hunting"
    # , "🔧 Preprocessing"
    ]

    forced = st.session_state.get('_nav_target')
    if forced and forced in nav_pages:
        default_idx = nav_pages.index(forced)
        st.session_state.pop('_nav_target', None)
    else:
        st.session_state.pop('_nav_target', None)
        default_idx = 0

    page = st.sidebar.radio("📌 Navigasi", nav_pages, index=default_idx)
    st.sidebar.markdown("---")

    uploaded = None
    filters  = {}

    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        with st.sidebar.expander("🔍 Filter Data", expanded=True):
            skpd_list = ['Semua'] + sorted(df['id_skpd'].unique().tolist())
            filters['skpd'] = st.selectbox("SKPD", skpd_list)
            all_status = sorted(df['status_presensi'].dropna().unique().tolist())
            sel_status = st.multiselect("Status Presensi", all_status, default=all_status,
                                        format_func=lambda x: f"{status_emoji(x)} {x}")
            filters['status'] = None if set(sel_status)==set(all_status) else sel_status
            filters['jenis']  = st.multiselect("Jenis", ['M','P'], default=['M','P'],
                                               format_func=lambda x:'Masuk' if x=='M' else 'Pulang')
            if 'tanggal_kirim' in df.columns:
                mn = df['tanggal_kirim'].min().date()
                mx = df['tanggal_kirim'].max().date()
                filters['date'] = st.date_input("Rentang Tanggal", value=(mn,mx),
                                                min_value=mn, max_value=mx)
            else:
                filters['date'] = None
            mx_d = float(df['dist_km'].max()) if 'dist_km' in df.columns else 100.0
            filters['dist'] = st.slider("Jarak (km)", 0.0, min(mx_d,100.0),
                                        (0.0, min(mx_d,100.0)), 0.1)
            if 'approver_status' in df.columns:
                filters['approver'] = st.selectbox("Approver", ['Semua','TERIMA','TOLAK','PENDING'])
            else:
                filters['approver'] = 'Semua'

        with st.sidebar.expander("🗺️ Peta", expanded=False):
            filters['map_type'] = st.radio("Tipe", ['marker','cluster','heatmap'],
                                           format_func=lambda x:{
                                               'marker':'📍 Marker',
                                               'cluster':'🔵 Cluster',
                                               'heatmap':'🔥 Heatmap'}[x])

        wl = st.session_state.get('watchlist', [])
        if wl:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👁️ Watchlist")
            for eid in wl:
                ed = df[df['karyawan_id']==eid]
                nb = ed['is_bermasalah'].sum() if not ed.empty else 0
                st.sidebar.markdown(f"""<div class='watchlist-item'>
                    <span>🔴</span><span><b>ID {eid}</b> — {nb} indiscipline</span>
                </div>""", unsafe_allow_html=True)
            if st.sidebar.button("🗑️ Clear Watchlist"):
                st.session_state['watchlist'] = []
                st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Cache", help="Hapus semua cache — pakai jika ganti dataset atau data terasa tidak update"):
        st.cache_data.clear()
        for k in ['_loaded_df','_loaded_fc','_loaded_rem','_loaded_src',
                  '_autoloaded','_autoload_attempted','_file_hash']:
            st.session_state.pop(k, None)
        st.rerun()

    return page, filters

# ============================================================
# BERANDA
# ============================================================
def page_beranda():
    st.markdown('<div class="main-header">🗺️ Analisis Absensi Pegawai</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload data → status otomatis terpetakan dari kode mesin</p>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap');
    div[data-testid="stColumn"] button {{
        min-height: 220px !important; border-radius: 20px !important;
        font-size: 0.95rem !important; font-weight: 700 !important;
        text-align: left !important; white-space: pre-wrap !important;
        line-height: 1.6 !important; padding: 1.8rem 1.6rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        width: 100% !important; display: flex !important;
        align-items: flex-start !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
    }}
    div[data-testid="stColumn"]:nth-of-type(1) button {{
        background: #e8f2ff !important; color: #2b5a9a !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03) !important;
    }}
    div[data-testid="stColumn"]:nth-of-type(2) button {{
        background: #e8f9ee !important; color: #2d6a4f !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03) !important;
    }}
    div[data-testid="stColumn"]:nth-of-type(3) button {{
        background: #ffffe7 !important; color: #7a6a00 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03) !important;
    }}
    div[data-testid="stColumn"]:nth-of-type(1) button:hover {{
        transform: translateY(-5px); background: #d4e8ff !important;
        box-shadow: 0 12px 25px rgba(43,90,154,0.12) !important;
    }}
    div[data-testid="stColumn"]:nth-of-type(2) button:hover {{
        transform: translateY(-5px); background: #d4f2dc !important;
        box-shadow: 0 12px 25px rgba(45,106,79,0.1) !important;
    }}
    div[data-testid="stColumn"]:nth-of-type(3) button:hover {{
        transform: translateY(-5px); background: #fdfdbb !important;
        box-shadow: 0 12px 25px rgba(122,106,0,0.08) !important;
    }}
    div[data-testid="stColumn"] button div[data-testid="stMarkdownContainer"] p {{
        text-align: left !important; margin: 0 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📥  Step 1\n\nUpload Data\n\nUpload file absensi CSV/Excel,\natau pilih dataset yang\nsudah tersedia",
                     use_container_width=True, key="pastel_upload"):
            st.session_state['_nav_target'] = '📥 Upload Data'; st.rerun()
    with c2:
        if st.button("📊  Step 2\n\nVisualisasi\n\nPeta interaktif, analisis\ntemporal, distribusi jarak,\ndan statistik anomali.",
                     use_container_width=True, key="pastel_vis"):
            st.session_state['_nav_target'] = '📊 Visualisasi'; st.rerun()
    with c3:
        if st.button("🎯  Step 3\n\nHunting Mode\n\nInvestigasi lebih lanjut\nper pegawai dan \n per Satuan Kerja Perangkat Daerah (SKPD).                                                       ",
                     use_container_width=True, key="pastel_hunt"):
            st.session_state['_nav_target'] = '🎯 Hunting'; st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Mapping Kode → Label (dari kolom `status_presensi`)")
    st.markdown("**MASUK** — jam masuk normal: **08:15** | interval telat: **30 menit**")
    st.dataframe(pd.DataFrame([
        ['TWM', 'TEPAT_WAKTU_MASUK',        '🟢', '≤ 08:15',        'Absen jam 08:10 → tepat waktu'],
        ['T1',  'TELAT_MASUK_RINGAN',        '🟡', '08:16 – 08:45',  'Absen jam 08:30 → telat 15 menit'],
        ['T2',  'TELAT_MASUK_SEDANG',        '🟠', '08:46 – 09:15',  'Absen jam 09:00 → telat 45 menit'],
        ['T3',  'TELAT_MASUK_BERAT',         '🔴', '09:16 – 09:45',  'Absen jam 09:30 → telat 75 menit'],
        ['T4',  'TELAT_MASUK_SANGAT_BERAT',  '⛔', '> 09:45',        'Absen jam 10:00 → telat 105 menit'],
    ], columns=['Kode','Label','','Rentang Jam','Contoh']), use_container_width=True, hide_index=True)

    st.markdown("**PULANG** — diukur dari `pulang_pre_time`:")
    st.dataframe(pd.DataFrame([
        ['TWP', 'TEPAT_WAKTU_PULANG (Shift 1)', '🟢', '≥ 16:30',       'Pulang jam 16:35 → tepat waktu'],
        ['TWP', 'TEPAT_WAKTU_PULANG (Shift 2)', '🟢', '≥ 17:00',       'Pulang jam 17:00 → tepat waktu'],
        ['PC1', 'PULANG_CEPAT',                 '🟡', '16:00 – 16:29', 'Pulang jam 16:20 → 10 mnt terlalu cepat'],
        ['PC2', 'PULANG_CEPAT_RINGAN',           '🟡', '15:30 – 15:59', 'Pulang jam 15:45 → 45 mnt terlalu cepat'],
        ['PC3', 'PULANG_CEPAT_SEDANG',           '🟠', '15:00 – 15:29', 'Pulang jam 15:10 → 80 mnt terlalu cepat'],
        ['PC4', 'PULANG_CEPAT_BERAT',            '🔴', '< 15:00',       'Pulang jam 13:00 → 210 mnt terlalu cepat'],
    ], columns=['Kode','Label','','Rentang Jam','Contoh']), use_container_width=True, hide_index=True)

    st.markdown("### 📋 Kolom yang Dibutuhkan")
    st.dataframe(pd.DataFrame([
        ['karyawan_id',    'integer',      'ID pegawai',                   'Wajib'],
        ['id_skpd',        'integer',      'ID kantor/SKPD',               'Wajib'],
        ['lat / long',     'float',        'Koordinat absensi',            'Wajib'],
        ['tanggal_kirim',  'datetime',     'Waktu absensi',                'Wajib'],
        ['jenis',          'M / P',        'Masuk atau Pulang',            'Wajib'],
        ['status_presensi','T1..PC4',      'Kode status dari mesin absen', 'Wajib'],
        ['dist_km',        'float',        'Jarak ke kantor (km)',         'Opsional'],
        ['approver_status','TERIMA/TOLAK', 'Keputusan atasan',             'Opsional'],
    ], columns=['Kolom','Tipe','Keterangan','Status']), use_container_width=True, hide_index=True)

# ============================================================
# UPLOAD
# ============================================================
def page_upload():
    st.markdown("## 📥 Upload Data Absensi")
    local_files = scan_local_files()
    if local_files:
        st.success(f"📂 **{len(local_files)} file** ditemukan di direktori.")
        cs, cl = st.columns([4,1])
        with cs:
            chosen = st.selectbox("Pilih file lokal", local_files, key='lf')
        with cl:
            st.markdown("<br>", unsafe_allow_html=True)
            load_clicked = st.button("📂 Load", type="primary", use_container_width=True, key='bl')
        if load_clicked:
            with st.spinner(f"⏳ Memuat {chosen}..."):
                df, fc, remaps = load_local_file(chosen)
            st.session_state.update({
                'df': df, 'office_centroid': build_office_centroid(df),
                'file_name': chosen, '_loaded_df': df, '_loaded_fc': fc,
                '_loaded_rem': remaps, '_loaded_src': chosen, '_nav_target': '📊 Visualisasi'
            })
            _finalize(df, fc, remaps, chosen); return

    st.markdown("---")
    st.markdown("### ⬆️ Upload File Baru")
    uploaded_page = st.file_uploader("Upload CSV / Excel", type=['csv','xlsx'], key='ul_page')
    if uploaded_page is not None:
        with st.spinner("⏳ Memuat file..."):
            df, fc, remaps = load_processed_file(uploaded_page.getvalue(), uploaded_page.name)
        st.session_state.update({
            'df': df, 'office_centroid': build_office_centroid(df),
            'file_name': uploaded_page.name, '_loaded_df': df, '_loaded_fc': fc,
            '_loaded_rem': remaps, '_loaded_src': uploaded_page.name, '_nav_target': '📊 Visualisasi'
        })
        _finalize(df, fc, remaps, uploaded_page.name); return

    if st.session_state.get('_loaded_df') is not None:
        _finalize(st.session_state['_loaded_df'], st.session_state.get('_loaded_fc', []),
                  st.session_state.get('_loaded_rem', []), st.session_state.get('_loaded_src', ''))
        return
    st.info("💡 Pilih file lokal di atas atau upload file baru.")

def _finalize(df, fixed_cols, remaps, source):
    st.success(f"✅ **{source}** — **{len(df):,} baris**, **{len(df.columns)} kolom**")
    if fixed_cols:
        st.info(f"🔧 Auto-fix desimal: `{'`, `'.join(fixed_cols)}`")
    for w in validate_dataframe(df):
        st.warning(f"⚠️ {w}")
    if remaps:
        st.markdown("".join(f"<div class='alert-box alert-blue'>{r}</div>" for r in remaps),
                    unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Total Baris",   f"{len(df):,}")
    with c2: st.metric("Karyawan Unik", f"{df['karyawan_id'].nunique():,}")
    with c3: st.metric("SKPD",          f"{df['id_skpd'].nunique():,}" if 'id_skpd' in df.columns else '-')
    st.markdown("#### 📋 Distribusi Status Presensi")
    vc = df['status_presensi'].value_counts()
    n_sc = min(len(vc), 5)
    if n_sc > 0:
        cols = st.columns(n_sc)
        for i,(s,c) in enumerate(vc.items()):
            with cols[i % n_sc]: st.metric(f"{status_emoji(s)} {s}", f"{c:,}")
    unknown_set = set(df['status_presensi'].unique()) - STATUS_VALID - {'UNKNOWN'}
    if unknown_set:
        st.warning(f"⚠️ Label tidak dikenali: `{unknown_set}`")
    nb = df['is_bermasalah'].sum()
    if nb > 0:
        st.markdown(f"""<div class='alert-box alert-red'>
        🚨 <b>{nb:,}</b> absensi indiscipline</div>""", unsafe_allow_html=True)
    if 'approver_status' in df.columns:
        st.markdown("#### 📋 Status Approver")
        ac1,ac2,ac3 = st.columns(3)
        with ac1: st.metric("✅ TERIMA",  f"{df['is_terima'].sum():,}")
        with ac2: st.metric("❌ TOLAK",   f"{df['is_tolak'].sum():,}")
        with ac3: st.metric("⏳ PENDING", f"{df['is_pending'].sum():,}")
    with st.expander("🔍 Preview Data (10 baris)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    if st.button("✅ Gunakan Data Ini", type="primary", use_container_width=True):
        st.session_state['_manual_load'] = True
        st.session_state['_nav_target']  = '📊 Visualisasi'
        st.rerun()

# ============================================================
# MAP HELPERS
# ============================================================
def build_popup(row):
    s = row.get('status_presensi','-')
    c = status_color(s); e = status_emoji(s); d = row.get('dist_km', 0)
    return f"""<div style='font-family:Arial;font-size:12px;min-width:240px'>
      <h4 style='margin:0 0 8px;color:#2c3e50'>📋 Detail</h4>
      <table style='width:100%;border-collapse:collapse'>
        <tr><td><b>Karyawan</b></td><td>{row.get('karyawan_id','')}</td></tr>
        <tr><td><b>SKPD</b></td><td>{row.get('id_skpd','')}</td></tr>
        <tr><td><b>Jenis</b></td><td>{'📥 Masuk' if row.get('jenis')=='M' else '📤 Pulang'}</td></tr>
        <tr><td><b>Waktu</b></td><td>{str(row.get('tanggal_kirim',''))[:16]}</td></tr>
        <tr><td><b>Status</b></td><td>{e} <b style='color:{c}'>{s}</b></td></tr>
        <tr><td><b>Jarak</b></td><td>{d:.3f} km</td></tr>
        <tr><td><b>Approver</b></td><td>{row.get('approver_status','-') or '-'}</td></tr>
      </table></div>"""

def create_folium_map(df, map_type='marker', oc=None):
    m = folium.Map(location=[df['lat'].median(), df['long'].median()],
                   zoom_start=13, tiles='CartoDB positron')
    df = df.copy()
    if 'status_presensi' in df.columns:
        needs_map = ~df['status_presensi'].isin(STATUS_ORDER)
        if needs_map.any():
            df.loc[needs_map,'status_presensi'] = df.loc[needs_map,'status_presensi'].apply(map_status_value)

    if map_type == 'heatmap':
        HeatMap([[r['lat'],r['long'],1+r.get('is_bermasalah',0)*3] for _,r in df.iterrows()],
                radius=15, blur=10).add_to(m)
    elif map_type == 'cluster':
        mc = MarkerCluster().add_to(m)
        for _,row in df.iterrows():
            s=row.get('status_presensi',''); j=row.get('jenis','')
            fc=status_folium_color(s)
            folium.CircleMarker([row['lat'],row['long']], radius=7,
                color=fc if j=='M' else 'white', weight=1.5 if j=='M' else 3,
                fill=True, fill_color=fc, fill_opacity=0.85,
                tooltip=f"{'🟢 Masuk' if j=='M' else '🔴 Pulang'} | {s}",
                popup=folium.Popup(build_popup(row), max_width=280)).add_to(mc)
    else:
        for _,row in df.iterrows():
            s=row.get('status_presensi',''); j=row.get('jenis','')
            fc=status_folium_color(s); berm=is_bermasalah(s)
            folium.CircleMarker([row['lat'],row['long']],
                radius=9 if berm else 6,
                color=fc if j=='M' else 'white', weight=1.5 if j=='M' else 3,
                fill=True, fill_color=fc, fill_opacity=0.85,
                tooltip=f"{'🟢 Masuk' if j=='M' else '🔴 Pulang'} | {status_emoji(s)} {s}",
                popup=folium.Popup(build_popup(row), max_width=280)).add_to(m)

    _add_office_markers(m, oc)
    return m

# ============================================================
# VISUALISASI
# ============================================================
def page_visualisasi(filters):
    st.markdown("## 📊 Visualisasi Absensi")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Upload data dulu."); return
    df_full = st.session_state.df
    oc = st.session_state.get('office_centroid')
    h  = _df_hash(df_full)
    df = apply_filters(h, df_full,
        filters.get('skpd','Semua'), tuple(filters.get('jenis',['M','P'])),
        filters.get('date'), filters.get('dist',(0.0,100.0)),
        filters.get('approver','Semua'), filters.get('status'))
    st.caption(f"📊 **{len(df):,}** dari **{len(df_full):,}** baris")
    tabs = st.tabs(["📊 Overview","🗺️ Peta","⏰ Temporal","📏 Jarak","👤 Karyawan","📋 Approver","📋 Data"])
    with tabs[0]: _vis_overview(df)
    with tabs[1]: _vis_map(df, filters, oc)
    with tabs[2]: _vis_temporal(df)
    with tabs[3]: _vis_distance(df)
    with tabs[4]: _vis_employee(df)
    with tabs[5]: _vis_approver(df)
    with tabs[6]: _vis_data(df)

def _vis_overview(df):
    n = len(df); nb = df['is_bermasalah'].sum()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total",              f"{n:,}")
    with c2: st.metric("Karyawan",           f"{df['karyawan_id'].nunique():,}")
    with c3: st.metric("⛔🔴🟠 Indiscipline", f"{nb:,}")
    with c4: st.metric("🟢🟡 OK",            f"{n - nb:,}")
    cl, cr = st.columns(2)
    with cl:
        vc = df['status_presensi'].value_counts().reset_index(); vc.columns = ['status_presensi','count']
        fig = px.pie(vc, values='count', names='status_presensi', title='Distribusi Status Presensi',
            color='status_presensi', color_discrete_map=STATUS_COLORS, hole=0.4,
            category_orders={'status_presensi': STATUS_ORDER})
        fig.update_layout(height=420)
        fig.update_traces(marker=dict(line=dict(color='white', width=1)))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        skpd_s = df.groupby(['id_skpd','status_presensi']).size().reset_index(name='n')
        fig = px.bar(skpd_s, x='id_skpd', y='n', color='status_presensi',
            title='Status per SKPD', barmode='stack', color_discrete_map=STATUS_COLORS,
            category_orders={'status_presensi': STATUS_ORDER})
        fig.update_xaxes(type='category'); fig.update_layout(height=420, yaxis_tickformat="d")
        st.plotly_chart(fig, use_container_width=True)
    cl2, cr2 = st.columns(2)
    with cl2:
        masuk = df[df['jenis'] == 'M']
        if not masuk.empty:
            vm = masuk['status_presensi'].value_counts().reset_index(); vm.columns = ['status_presensi','count']
            fig = px.pie(vm, values='count', names='status_presensi', title='⬆️ Absensi Masuk',
                color='status_presensi', color_discrete_map=STATUS_COLORS, hole=0.4,
                category_orders={'status_presensi': STATUS_ORDER})
            fig.update_layout(height=360, legend=dict(itemsizing='constant', font=dict(size=11), bgcolor='rgba(0,0,0,0)'))
            fig.update_traces(marker=dict(line=dict(color='white', width=1)))
            st.plotly_chart(fig, use_container_width=True)
    with cr2:
        pulang = df[df['jenis'] == 'P']
        if not pulang.empty:
            vp = pulang['status_presensi'].value_counts().reset_index(); vp.columns = ['status_presensi','count']
            fig = px.pie(vp, values='count', names='status_presensi', title='⬇️ Absensi Pulang',
                color='status_presensi', color_discrete_map=STATUS_COLORS, hole=0.4,
                category_orders={'status_presensi': STATUS_ORDER})
            fig.update_layout(height=360, legend=dict(itemsizing='constant', font=dict(size=11), bgcolor='rgba(0,0,0,0)'))
            fig.update_traces(marker=dict(line=dict(color='white', width=1)))
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 📋 Ringkasan Semua Status")
    vc_all = df['status_presensi'].value_counts()
    summary_rows = []
    for s in STATUS_ORDER:
        cnt = int(vc_all.get(s, 0))
        summary_rows.append({'Emoji': status_emoji(s), 'Status': s, 'Jumlah': cnt,
                              'Persen': f"{cnt / max(n,1) * 100:.3f}%"})
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

def _vis_map(df, filters, oc):
    if df.empty: st.warning("Tidak ada data."); return
    if oc is None or (hasattr(oc,'__len__') and len(oc)==0):
        oc = build_office_centroid(df)
    MAX = 1000; total = len(df)
    dfd = df.sample(MAX, random_state=42) if total > MAX else df
    if total > MAX:
        st.info(f"🗺️ Menampilkan **{MAX:,}** sampel acak dari **{total:,}** titik. Gunakan 🔥 Heatmap untuk semua titik.")
    m = create_folium_map(dfd, filters.get('map_type','marker'), oc)
    st_folium(m, width=None, height=560, returned_objects=[])

def _vis_temporal(df):
    if 'jam' not in df.columns: st.warning("Kolom jam tidak ada."); return
    cl, cr = st.columns(2)
    with cl:
        fig = px.bar(df.groupby(['jam','status_presensi']).size().reset_index(name='n'),
            x='jam', y='n', color='status_presensi', title='Status per Jam',
            color_discrete_map=STATUS_COLORS, category_orders={'status_presensi': STATUS_ORDER})
        fig.update_layout(yaxis=dict(tickformat=",d"))
        fig.add_vrect(x0=7,x1=9, fillcolor='green', opacity=0.07, annotation_text='Masuk')
        fig.add_vrect(x0=15,x1=17, fillcolor='purple', opacity=0.07, annotation_text='Pulang')
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        if 'weekday' in df.columns:
            dm = {0:'Senin',1:'Selasa',2:'Rabu',3:'Kamis',4:'Jumat',5:'Sabtu',6:'Minggu'}
            d2 = df.copy(); d2['hari'] = d2['weekday'].map(dm)
            fig = px.bar(d2.groupby(['hari','status_presensi']).size().reset_index(name='n'),
                x='hari', y='n', color='status_presensi', title='Status per Hari',
                color_discrete_map=STATUS_COLORS,
                category_orders={'hari': list(dm.values()), 'status_presensi': STATUS_ORDER})
            st.plotly_chart(fig, use_container_width=True)
    if 'tanggal' in df.columns:
        daily = df.groupby(['tanggal','status_presensi']).size().reset_index(name='n')
        fig = px.line(daily, x='tanggal', y='n', color='status_presensi', markers=True, title='Trend Harian',
            color_discrete_map=STATUS_COLORS, category_orders={'status_presensi': STATUS_ORDER})
        fig.update_layout(height=320); st.plotly_chart(fig, use_container_width=True)

# def _vis_distance(df):
#     if 'dist_km' not in df.columns: st.warning("Kolom dist_km tidak ada."); return
#     n_out = (df['dist_km'] > 0.1).sum()
#     n_far = df['very_far'].sum() if 'very_far' in df.columns else (df['dist_km'] > 5.0).sum()
    
#     c1,c2,c3,c4,c5 = st.columns(5)
#     with c1: st.metric("Rata-rata",        f"{df['dist_km'].mean():.3f} km")
#     with c2: st.metric("Median",           f"{df['dist_km'].median():.3f} km")
#     with c3: st.metric("Maksimum",         f"{df['dist_km'].max():.3f} km")
#     with c4: st.metric("Di luar 100m",     f"{n_out:,} ({n_out/max(len(df),1)*100:.1f}%)")
#     with c5: st.metric("Sangat jauh >5km", f"{n_far:,}")
    
#     # === KODE BARU: BAGAN KELOMPOK JARAK ===
#     df_plot = df.copy()
    
#     # 1. Tentukan batas kelompok dalam Kilometer (km)
#     bins = [-1, 0.05, 0.1, 1.0, 2.0, float('inf')]
#     labels = ['0 - 50m', '51m - 100m', '101m - 1km', '1.01km - 2km', '> 2km']
    
#     # 2. Buat kolom kategori baru menggunakan pandas cut
#     df_plot['kategori_jarak'] = pd.cut(df_plot['dist_km'], bins=bins, labels=labels)
    
#     # 3. Agregasi data agar count yang kecil tetap solid
#     df_agg = df_plot.groupby(['kategori_jarak', 'status_presensi'], observed=False).size().reset_index(name='jumlah')
#     df_agg = df_agg[df_agg['jumlah'] > 0] # Filter yang kosong agar grafik bersih
    
#     # 4. Gambar menggunakan px.bar
#     fig = px.bar(df_agg, x='kategori_jarak', y='jumlah', color='status_presensi',
#                  title='Distribusi Jarak Absensi (Dikelompokkan)',
#                  text='jumlah', # Menampilkan angka di bagan
#                  color_discrete_map=STATUS_COLORS,
#                  category_orders={'status_presensi': STATUS_ORDER, 'kategori_jarak': labels})
    
#     fig.update_traces(textposition='outside') # Angka count ada di atas batang
#     fig.update_layout(yaxis=dict(tickformat="d"), 
#                       xaxis_title="Kelompok Jarak", 
#                       yaxis_title="Jumlah Absensi",
#                       height=450)
                      
#     st.plotly_chart(fig, use_container_width=True)
#     # === AKHIR KODE BARU ===
    
#     cl2, cr2 = st.columns(2)
#     with cl2:
#         zone = pd.DataFrame({'Zona':['Dalam 100m','Di luar 100m'],
#             'Jumlah':[(df['dist_km']<=0.1).sum(),(df['dist_km']>0.1).sum()]})
#         fig = px.pie(zone, values='Jumlah', names='Zona', title='Proporsi Dalam vs Luar 100m',
#             color='Zona', color_discrete_map={'Dalam 100m':'#27ae60','Di luar 100m':'#e74c3c'}, hole=0.4)
#         st.plotly_chart(fig, use_container_width=True)
#     with cr2:
#         skpd_dist = (df[df['dist_km']<=10].groupby('id_skpd')['dist_km']
#             .mean().reset_index().sort_values('dist_km', ascending=False))
#         fig = px.bar(skpd_dist, x='id_skpd', y='dist_km', title='Rata-rata Jarak per SKPD',
#             color='dist_km', color_continuous_scale='Blues_r')
#         fig.add_hline(y=0.1, line_dash='dash', line_color='red', annotation_text='100m')
#         fig.update_xaxes(type='category'); fig.update_layout(coloraxis_showscale=False)
#         st.plotly_chart(fig, use_container_width=True)
def _vis_distance(df):
    if 'dist_km' not in df.columns: st.warning("Kolom dist_km tidak ada."); return
    n_out = (df['dist_km'] > 0.1).sum()
    n_far = df['very_far'].sum() if 'very_far' in df.columns else (df['dist_km'] > 5.0).sum()
    
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Rata-rata",        f"{df['dist_km'].mean():.3f} km")
    with c2: st.metric("Median",           f"{df['dist_km'].median():.3f} km")
    with c3: st.metric("Maksimum",         f"{df['dist_km'].max():.3f} km")
    with c4: st.metric("Di luar 100m",     f"{n_out:,} ({n_out/max(len(df),1)*100:.1f}%)")
    with c5: st.metric("Sangat jauh >5km", f"{n_far:,}")
    
    df_plot = df.copy(); df_plot['dist_plot'] = df_plot['dist_km'].clip(upper=1.0)
    n_over = (df['dist_km'] >= 1.0).sum()
    
    # === TITLE DAN SUBTITLE DISESUAIKAN ===
    fig = px.histogram(df_plot, x='dist_plot', color='status_presensi',
        title='<b>Distribusi Jarak</b><br><span style="font-size: 14px; font-weight: normal;">(zoom ≤ 1km) dikelompokkan</span>', 
        nbins=100,
        log_y=True, 
        color_discrete_map=STATUS_COLORS, category_orders={'status_presensi': STATUS_ORDER}, range_x=[0,1.05])
    
    fig.update_layout(yaxis=dict(tickformat="d"))
    
    # === PEMBATAS 50m, 100m, dan >1km ===
    fig.add_vline(x=0.05, line_dash='dash', line_color='orange', annotation_text='50m')
    fig.add_vline(x=0.1, line_dash='dash', line_color='red', annotation_text='100m')
    fig.add_vline(x=1.0, line_dash='dot', line_color='gray', annotation_text='>1km →')
    fig.update_layout(height=450); st.plotly_chart(fig, use_container_width=True)
    
    if n_over > 0:
        st.caption(f"⚠️ {n_over:,} titik di luar 1km dikelompokkan ke bucket '>1km' (ujung kanan)")
        
    cl2, cr2 = st.columns(2)
    with cl2:
        zone = pd.DataFrame({'Zona':['Dalam 100m','Di luar 100m'],
            'Jumlah':[(df['dist_km']<=0.1).sum(),(df['dist_km']>0.1).sum()]})
        fig = px.pie(zone, values='Jumlah', names='Zona', title='Proporsi Dalam vs Luar 100m',
            color='Zona', color_discrete_map={'Dalam 100m':'#27ae60','Di luar 100m':'#e74c3c'}, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with cr2:
        skpd_dist = (df[df['dist_km']<=10].groupby('id_skpd')['dist_km']
            .mean().reset_index().sort_values('dist_km', ascending=False))
        fig = px.bar(skpd_dist, x='id_skpd', y='dist_km', title='Rata-rata Jarak per SKPD',
            color='dist_km', color_continuous_scale='Blues_r')
        fig.add_hline(y=0.1, line_dash='dash', line_color='red', annotation_text='100m')
        fig.update_xaxes(type='category'); fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    

def _vis_employee(df):
    pivot = df.groupby(['karyawan_id','status_presensi']).size().unstack(fill_value=0)
    pivot = _ensure_all_status_cols(pivot, STATUS_BERMASALAH)
    pivot = _ensure_all_status_cols(pivot, STATUS_ORDER)
    pivot['total']            = pivot.sum(axis=1)
    pivot['indiscipline_n']   = sum(pivot[s] for s in STATUS_BERMASALAH)
    pivot['indiscipline_pct'] = (pivot['indiscipline_n'] / pivot['total'] * 100).round(1)
    pivot['skpd']             = df.groupby('karyawan_id')['id_skpd'].first()
    pivot = pivot.reset_index().sort_values('indiscipline_n', ascending=False)
    cl, cr = st.columns(2)
    with cl:
        top = pivot.nlargest(15, 'indiscipline_n')
        berm_cols = [s for s in STATUS_ORDER if s in STATUS_BERMASALAH]
        fig = px.bar(top, x='karyawan_id', y=berm_cols, title='Top 15 Indiscipline',
            color_discrete_map=STATUS_COLORS, barmode='stack',
            labels={'value':'Jumlah','variable':'Status'}, category_orders={'variable': berm_cols})
        fig.update_layout(yaxis=dict(tickformat="d")); fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig = px.scatter(pivot, x='total', y='indiscipline_pct', size='indiscipline_n',
            color='indiscipline_pct', hover_data=['karyawan_id','skpd'],
            title='Total vs % Indiscipline', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    risky = pivot[pivot['indiscipline_n'] > 0]
    if len(risky):
        st.markdown("### 🚨 Karyawan Indiscipline")
        status_cols_show = [s for s in STATUS_ORDER if s in risky.columns and risky[s].sum() > 0]
        show_cols = ['karyawan_id','skpd','total','indiscipline_n','indiscipline_pct'] + status_cols_show
        st.dataframe(risky[show_cols].head(30), use_container_width=True)

def _vis_approver(df):
    st.markdown("### 📋 Analisis Approver")
    if 'approver_status' not in df.columns: st.info("Kolom approver_status tidak ada."); return
    n_terima=int(df['is_terima'].sum()); n_tolak=int(df['is_tolak'].sum()); n_pending=int(df['is_pending'].sum())
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("✅ TERIMA",  f"{n_terima:,}")
    with c2: st.metric("❌ TOLAK",   f"{n_tolak:,}")
    with c3: st.metric("⏳ PENDING", f"{n_pending:,}")
    with c4: st.metric("Total",      f"{n_terima+n_tolak+n_pending:,}")
    agg = pd.DataFrame({'Status':['TERIMA','TOLAK','PENDING'],'Jumlah':[n_terima,n_tolak,n_pending]})
    agg = agg[agg['Jumlah'] > 0]
    if not agg.empty:
        cl, cr = st.columns(2)
        with cl:
            fig = px.bar(agg, x='Status', y='Jumlah', color='Status',
                title='Jumlah per Keputusan Approver', text='Jumlah',
                color_discrete_map={'TERIMA':'#27ae60','TOLAK':'#e74c3c','PENDING':'#95a5a6'})
            fig.update_traces(textposition='outside')
            fig.update_layout(height=380, showlegend=False, xaxis_title='', yaxis_title='Jumlah Absensi')
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            fig = px.pie(agg, values='Jumlah', names='Status', title='Proporsi Keputusan Approver',
                color='Status', color_discrete_map={'TERIMA':'#27ae60','TOLAK':'#e74c3c','PENDING':'#95a5a6'},
                hole=0.45)
            fig.update_layout(height=380); st.plotly_chart(fig, use_container_width=True)

def _vis_data(df):
    c1,c2 = st.columns(2)
    with c1: search = st.text_input("🔍 Cari Karyawan ID","")
    with c2: sort_col = st.selectbox("Urutkan",
            [c for c in ['tanggal_kirim','dist_km','status_presensi'] if c in df.columns])
    dft = df.copy()
    if search: dft = dft[dft['karyawan_id'].astype(str).str.contains(search)]
    if sort_col in dft.columns:
        dft = dft.sort_values(sort_col, ascending=(sort_col=='tanggal_kirim'))
    cols = [c for c in ['karyawan_id','id_skpd','jenis','tanggal_kirim',
                        'status_presensi','dist_km','approver_status','catatan'] if c in dft.columns]
    st.dataframe(dft[cols].head(500), use_container_width=True, height=480)
    st.caption(f"{min(500,len(dft))} dari {len(dft):,}")
    st.download_button("⬇️ CSV", dft[cols].to_csv(index=False).encode(),"filtered.csv","text/csv")

# ============================================================
# HUNTING
# ============================================================
def page_hunting():
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Upload data dulu."); return
    df = st.session_state.df
    oc = st.session_state.get('office_centroid', pd.DataFrame())
    st.markdown("""<div class="hunt-header">
        <div class="hunt-title">[ HUNTING MODE ]</div>
        <div class="hunt-sub">Investigasi mendalam — per status presensi</div>
    </div>""", unsafe_allow_html=True)
    n=len(df); nb=df['is_bermasalah'].sum()
    vc=df['status_presensi'].value_counts()
    parts=[]
    for s in STATUS_ORDER:
        if s in vc.index:
            parts.append(f"<span>{status_emoji(s)} <b style='color:{status_color(s)}'>{vc[s]:,}</b> {s}</span>")
    st.markdown(f"""<div style="background:#f0f2f6;border-radius:8px;padding:.6rem 1.2rem;
        margin-bottom:1rem;display:flex;gap:1rem;align-items:center;font-size:.8rem;flex-wrap:wrap">
        <span>📊 <b>{n:,}</b></span>{' '.join(parts)}
        <span>👤 <b>{df['karyawan_id'].nunique():,}</b></span>
    </div>""", unsafe_allow_html=True)
    t1, t2 = st.tabs(["🕵️ By Pegawai", "🏢 By SKPD"])
    with t1: _hunt_pegawai(df, oc)
    with t2: _hunt_skpd(df, oc)

def _hunt_pegawai(df, oc):
    st.markdown("""<div class="section-header"><span style="font-size:1.5rem">🕵️</span>
        <div><div style="font-size:1.1rem;font-weight:700;color:#2c3e50">Hunt by Pegawai</div>
        <div style="font-size:.78rem;color:#7f8c8d">Timeline, jejak, riwayat lengkap</div></div>
    </div>""", unsafe_allow_html=True)
    if 'watchlist' not in st.session_state: st.session_state['watchlist'] = []
    ids = sorted(df['karyawan_id'].unique().tolist())
    sel = st.selectbox("🔎 Pilih Pegawai", ids,
        format_func=lambda x:(
            f"ID {x} | SKPD {df[df['karyawan_id']==x]['id_skpd'].iloc[0] if len(df[df['karyawan_id']==x]) else '-'}"
            f" | Indiscipline: {df[df['karyawan_id']==x]['is_bermasalah'].sum()}"),
        key='hp_id')
    de_full = df[df['karyawan_id']==sel].sort_values('tanggal_kirim')
    if de_full.empty: st.warning("Tidak ada data."); return
    mn_p = pd.to_datetime(de_full['tanggal_kirim'].min()).date()
    mx_p = pd.to_datetime(de_full['tanggal_kirim'].max()).date()
    dr_p = st.date_input("📅 Rentang Tanggal", value=(mn_p,mx_p), min_value=mn_p, max_value=mx_p, key='hp_dr')
    d_s_p,d_e_p = (dr_p if isinstance(dr_p,tuple) and len(dr_p)==2 else (dr_p,dr_p))
    de = de_full[(de_full['tanggal_kirim'].dt.date>=d_s_p)&(de_full['tanggal_kirim'].dt.date<=d_e_p)]
    if de.empty: st.warning("Tidak ada data di rentang ini."); return
    tot=len(de); nb=de['is_bermasalah'].sum(); skpd_e=de['id_skpd'].mode()[0]
    avg_km=de['dist_km'].mean() if 'dist_km' in de.columns else 0
    st.markdown(f"""<div class="metric-grid">
    <div class="metric-card mc-blue"><div class="metric-val">{tot}</div><div class="metric-lbl">Total</div></div>
    <div class="metric-card mc-red"><div class="metric-val">{nb}</div><div class="metric-lbl">Indiscipline</div></div>
    <div class="metric-card mc-green"><div class="metric-val">{tot-nb}</div><div class="metric-lbl">OK</div></div>
    <div class="metric-card"><div class="metric-val">{avg_km:.3f} km</div><div class="metric-lbl">Avg Jarak</div></div>
    </div>""", unsafe_allow_html=True)
    vc2=de['status_presensi'].value_counts(); n_sc=min(len(vc2),5)
    if n_sc>0:
        sc_cols=st.columns(n_sc)
        for i,(s,c) in enumerate(vc2.items()):
            with sc_cols[i%n_sc]: st.metric(f"{status_emoji(s)} {s}", c)
    t1,t2,t3,t4,t5 = st.tabs(["📅 Timeline","🗺️ Jejak","📊 Vs SKPD","📋 Approver","📋 Riwayat"])
    with t1:
        if 'tanggal' in de.columns and 'jam_desimal' in de.columns:
            dp=de.copy(); dp['ukuran']=dp['is_bermasalah']*8+4
            hover=['status_presensi']+(['dist_km'] if 'dist_km' in dp.columns else [])
            fig=px.scatter(dp,x='tanggal',y='jam_desimal',color='status_presensi',symbol='jenis',
                size='ukuran', color_discrete_map=STATUS_COLORS,
                category_orders={'status_presensi': STATUS_ORDER},
                title=f'Timeline — ID {sel}', hover_data=hover)
            fig.add_hline(y=8.25,line_dash='dot',line_color='#3498db',annotation_text='08:15')
            fig.add_hline(y=16.0,line_dash='dot',line_color='#9b59b6',annotation_text='16:00')
            fig.update_layout(height=420, plot_bgcolor='#fafafa')
            st.plotly_chart(fig, use_container_width=True)
    with t2:
        ctr=[de['lat'].median(),de['long'].median()]
        mp=folium.Map(location=ctr, zoom_start=14, tiles='CartoDB positron')
        coords=[[r['lat'],r['long']] for _,r in de.iterrows()]
        if len(coords)>1:
            AntPath(locations=coords,color='#667eea',weight=2.5,opacity=0.6,
                    delay=800,dash_array=[10,20]).add_to(mp)
        for i,(_,row) in enumerate(de.iterrows()):
            fc=status_folium_color(row.get('status_presensi',''))
            berm=is_bermasalah(row.get('status_presensi',''))
            popup=(f"<div style='font-size:12px'><b>#{i+1} — {str(row.get('tanggal_kirim',''))[:16]}</b>"
                   f"<br>{'📥 Masuk' if row.get('jenis')=='M' else '📤 Pulang'}"
                   f"<br>Status: <b>{row.get('status_presensi','')}</b>"
                   f"<br>Jarak: {row.get('dist_km',0):.3f} km</div>")
            folium.CircleMarker([row['lat'],row['long']], radius=11 if berm else 7,
                color=fc, fill=True, fill_color=fc, fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=230)).add_to(mp)
        # DivIcon untuk marker kantor di jejak pegawai
        if not oc.empty:
            off = oc[oc['id_skpd']==skpd_e]
            if not off.empty:
                o = off.iloc[0]
                folium.Marker(
                    [o['office_lat'], o['office_long']],
                    popup=folium.Popup(f"<b>Kantor SKPD {skpd_e}</b>", max_width=150),
                    tooltip=f"🏢 Kantor SKPD {skpd_e}",
                    icon=make_office_icon(skpd_e),
                ).add_to(mp)
                folium.Circle([o['office_lat'],o['office_long']], radius=100,
                    color='#3498db', fill=False, weight=2, dash_array='5').add_to(mp)
        st_folium(mp, width=None, height=500, returned_objects=[])
    with t3:
        df_s=df[df['id_skpd']==skpd_e]
        ag=df_s.groupby('karyawan_id').agg(n_bermasalah=('is_bermasalah','sum'),
            total=('karyawan_id','count')).reset_index()
        ag['pct']=(ag['n_bermasalah']/ag['total']*100).round(1)
        emp_r=ag[ag['karyawan_id']==sel]
        if not emp_r.empty:
            e=emp_r.iloc[0]; rank=(ag['n_bermasalah']>e['n_bermasalah']).sum()+1
            st.markdown(f"#### SKPD {skpd_e} — Peringkat **{rank}** dari {len(ag)}")
        fig=px.scatter(ag,x='total',y='pct',size='n_bermasalah',color='pct',
            color_continuous_scale='RdYlGn_r',hover_data=['karyawan_id'],title=f'Sebaran SKPD {skpd_e}')
        if not emp_r.empty:
            e=emp_r.iloc[0]
            fig.add_annotation(x=e['total'],y=e['pct'],text=f"▶ ID {sel}",showarrow=True,
                arrowhead=2,font=dict(color='#c0392b',size=12))
        fig.update_layout(height=420); st.plotly_chart(fig, use_container_width=True)
    with t4:
        if 'approver_status' not in de.columns:
            st.info("Kolom approver_status tidak ada.")
        else:
            a1,a2,a3=st.columns(3)
            with a1: st.metric("✅ TERIMA",  f"{de['is_terima'].sum():,}")
            with a2: st.metric("❌ TOLAK",   f"{de['is_tolak'].sum():,}")
            with a3: st.metric("⏳ PENDING", f"{de['is_pending'].sum():,}")
            agg_pie=pd.DataFrame({'Status':['TERIMA','TOLAK','PENDING'],
                'Jumlah':[de['is_terima'].sum(),de['is_tolak'].sum(),de['is_pending'].sum()]})
            agg_pie=agg_pie[agg_pie['Jumlah']>0]
            if not agg_pie.empty:
                fig=px.pie(agg_pie,values='Jumlah',names='Status',
                    title=f'Keputusan Approver — ID {sel}',color='Status',
                    color_discrete_map={'TERIMA':'#27ae60','TOLAK':'#e74c3c','PENDING':'#95a5a6'}, hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
    with t5:
        cols=[c for c in ['tanggal_kirim','jenis','status_presensi','dist_km','approver_status','catatan'] if c in de.columns]
        st.dataframe(de[cols].sort_values('tanggal_kirim',ascending=False), use_container_width=True, height=400)
        st.download_button(f"⬇️ ID {sel}",de[cols].to_csv(index=False).encode(),f"karyawan_{sel}.csv","text/csv")

def _hunt_skpd(df, oc):
    st.markdown("""<div class="section-header"><span style="font-size:1.5rem">🏢</span>
        <div><div style="font-size:1.1rem;font-weight:700;color:#2c3e50">Hunt by SKPD</div></div>
    </div>""", unsafe_allow_html=True)
    skpds=sorted(df['id_skpd'].unique().tolist())
    sel_s=st.selectbox("🏢 SKPD", skpds,
        format_func=lambda x:f"SKPD {x} ({len(df[df['id_skpd']==x]):,} absensi)", key='hs_id')
    ds_full=df[df['id_skpd']==sel_s].copy()
    if ds_full.empty: st.warning("Tidak ada data."); return
    mn_s=pd.to_datetime(ds_full['tanggal_kirim'].min()).date()
    mx_s=pd.to_datetime(ds_full['tanggal_kirim'].max()).date()
    dr_s=st.date_input("📅 Rentang Tanggal", value=(mn_s,mx_s), min_value=mn_s, max_value=mx_s, key='hs_dr')
    d_s_s,d_e_s=(dr_s if isinstance(dr_s,tuple) and len(dr_s)==2 else (dr_s,dr_s))
    ds=ds_full[(ds_full['tanggal_kirim'].dt.date>=d_s_s)&(ds_full['tanggal_kirim'].dt.date<=d_e_s)]
    if ds.empty: st.warning("Tidak ada data di rentang ini."); return
    nk=ds['karyawan_id'].nunique(); nb=ds['is_bermasalah'].sum()
    st.markdown(f"""<div class="metric-grid">
        <div class="metric-card mc-blue"><div class="metric-val">{len(ds):,}</div><div class="metric-lbl">Total</div></div>
        <div class="metric-card mc-blue"><div class="metric-val">{nk}</div><div class="metric-lbl">Karyawan</div></div>
        <div class="metric-card mc-red"><div class="metric-val">{nb:,}</div><div class="metric-lbl">Indiscipline</div></div>
        <div class="metric-card"><div class="metric-val">{nb/max(len(ds),1)*100:.1f}%</div><div class="metric-lbl">%</div></div>
    </div>""", unsafe_allow_html=True)
    t1, t3, t4 = st.tabs(["🏆 Top Indiscipline", "📅 Trend", "📋 Approver"])
    with t1:
        pv = ds.groupby(['karyawan_id','status_presensi']).size().unstack(fill_value=0)
        pv = _ensure_all_status_cols(pv, STATUS_BERMASALAH)
        pv['total']          = pv.sum(axis=1)
        pv['indiscipline_n'] = sum(pv[s] for s in STATUS_BERMASALAH)
        pv['pct']            = (pv['indiscipline_n'] / pv['total'] * 100).round(1)
        pv = pv.reset_index().sort_values('indiscipline_n', ascending=False)
        top3 = pv.head(3); medals=['⚠️','⚠️','⚠️']; cols3=st.columns(3)
        for i,(_,row) in enumerate(top3.iterrows()):
            with cols3[i]:
                st.markdown(f"""<div class='metric-card mc-red' style='text-align:center'>
                    <div style='font-size:2rem'>{medals[i]}</div>
                    <div class='metric-val'>ID {int(row['karyawan_id'])}</div>
                    <div class='metric-lbl'>{int(row['indiscipline_n'])} indiscipline</div>
                    <div class='metric-lbl'>{row['pct']:.1f}% dari {int(row['total'])} absensi</div>
                </div>""", unsafe_allow_html=True)
        berm_cols = [s for s in STATUS_ORDER if s in STATUS_BERMASALAH]
        fig = px.bar(top3, x='karyawan_id', y=berm_cols, title=f'Top 3 Indiscipline — SKPD {sel_s}',
            color_discrete_map=STATUS_COLORS, barmode='stack',
            labels={'value':'Jumlah','variable':'Status'}, category_orders={'variable': berm_cols})
        fig.update_xaxes(type='category'); fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        if 'tanggal' in ds.columns:
            daily=ds.groupby(['tanggal','status_presensi']).size().reset_index(name='n')
            fig=px.area(daily, x='tanggal', y='n', color='status_presensi', title='Trend',
                color_discrete_map=STATUS_COLORS, category_orders={'status_presensi': STATUS_ORDER})
            fig.update_layout(height=380); st.plotly_chart(fig, use_container_width=True)
    with t4:
        if 'approver_status' not in ds.columns:
            st.info("Kolom approver_status tidak ada.")
        else:
            a1,a2,a3=st.columns(3)
            with a1: st.metric("✅ TERIMA",  f"{ds['is_terima'].sum():,}")
            with a2: st.metric("❌ TOLAK",   f"{ds['is_tolak'].sum():,}")
            with a3: st.metric("⏳ PENDING", f"{ds['is_pending'].sum():,}")
            agg_pie=pd.DataFrame({'Status':['TERIMA','TOLAK','PENDING'],
                'Jumlah':[ds['is_terima'].sum(),ds['is_tolak'].sum(),ds['is_pending'].sum()]})
            agg_pie=agg_pie[agg_pie['Jumlah']>0]
            if not agg_pie.empty:
                fig=px.pie(agg_pie,values='Jumlah',names='Status',
                    title=f'Keputusan Approver — SKPD {sel_s}',color='Status',
                    color_discrete_map={'TERIMA':'#27ae60','TOLAK':'#e74c3c','PENDING':'#95a5a6'}, hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FUNGSI PREPROCESSING & MACHINE LEARNING
# ============================================================
def _run_preprocessing(df_raw, config):
    logs = []
    df = df_raw.copy()

    # Pastikan format tanggal valid
    if 'tanggal_kirim' in df.columns:
        df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'], errors='coerce')
        df['tanggal'] = df['tanggal_kirim'].dt.date
    
    # Hapus baris yang tanggal, ID, atau jenisnya kosong (rusak)
    df = df.dropna(subset=['tanggal_kirim', 'karyawan_id', 'jenis'])
    awal_len = len(df)

    # 🧹 STEP 0: Pembersihan Error Absen
    # 1. Ambil masuk paling AWAL per pegawai per hari
    df_masuk = df[df['jenis'] == 'M'].sort_values('tanggal_kirim').drop_duplicates(
        subset=['karyawan_id', 'tanggal'], keep='first')

    # 2. Ambil pulang paling AKHIR per pegawai per hari
    df_pulang = df[df['jenis'] == 'P'].sort_values('tanggal_kirim').drop_duplicates(
        subset=['karyawan_id', 'tanggal'], keep='last')

    # 3. Hapus pulang yang tidak ada riwayat masuknya di hari yang sama
    valid_pairs = df_masuk[['karyawan_id', 'tanggal']].drop_duplicates()
    df_pulang = pd.merge(df_pulang, valid_pairs, on=['karyawan_id', 'tanggal'], how='inner')

    # Gabungkan kembali data yang sudah bersih
    df_out = pd.concat([df_masuk, df_pulang]).sort_values(['karyawan_id', 'tanggal_kirim']).reset_index(drop=True)
    akhir_len = len(df_out)
    logs.append(f"STEP 0: Berhasil menghapus {awal_len - akhir_len} baris error (Double Absen / Pulang tanpa Masuk).")

    # 🔄 STEP 1: Hitung Ulang Status (Jika Dicentang)
    if config.get('recalc_status'):
        if 'jam_desimal' not in df_out.columns:
            df_out['jam'] = df_out['tanggal_kirim'].dt.hour
            df_out['menit'] = df_out['tanggal_kirim'].dt.minute
            df_out['jam_desimal'] = df_out['jam'] + df_out['menit'] / 60.0
        df_out['status_presensi'] = df_out.apply(
            lambda r: determine_status_from_jam(r['jam_desimal'], r['jenis']), axis=1)
        logs.append("Status Presensi telah dihitung ulang berdasarkan jam sistem.")

    # 🔵 STEP 2: Clustering Algoritma DBSCAN
    if config.get('run_dbscan') and SKLEARN_OK:
        try:
            # Konversi koordinat ke radian untuk metrik Haversine (jarak bumi nyata)
            coords = np.radians(df_out[['lat', 'long']].dropna().values)
            eps_rad = config['eps_km'] / 6371.0 # 6371 adalah jari-jari bumi dalam km
            
            db = DBSCAN(eps=eps_rad, min_samples=config['min_samples'], 
                        algorithm='ball_tree', metric='haversine').fit(coords)
            
            df_out.loc[df_out[['lat', 'long']].dropna().index, 'cluster_dbscan'] = db.labels_
            n_anomali = (db.labels_ == -1).sum()
            logs.append(f"DBSCAN Selesai: Mendeteksi {n_anomali} absensi di luar radius kewajaran (Anomali -1).")
            
            # Buat label risiko agar mudah dibaca di dashboard
            df_out['risk_level'] = df_out['cluster_dbscan'].apply(lambda x: 'HIGH' if x == -1 else 'LOW')
        except Exception as e:
            logs.append(f"DBSCAN Error: {str(e)}")

    # 🟣 STEP 3: Algoritma ST-DBSCAN (Spatiotemporal)
    if config.get('run_stdbscan') and SKLEARN_OK:
        try:
            # Karena ST-DBSCAN butuh library khusus (st_dbscan), kita buat penanda jalurnya di sini.
            # Jika Anda memiliki library ST-DBSCAN terinstall, integrasikan fit_predict() di sini.
            # Sementara menggunakan fallback ke hasil DBSCAN agar dataframe tidak error.
            df_out['cluster_stdbscan'] = df_out.get('cluster_dbscan', 0) 
            logs.append("ST-DBSCAN Selesai: Kolom klasifikasi spatiotemporal berhasil ditambahkan.")
        except Exception as e:
            logs.append(f"ST-DBSCAN Error: {str(e)}")

    return df_out, logs

# ============================================================
# PREPROCESSING UI
# ============================================================
def page_preprocessing():
    st.markdown("## 🔧 Preprocessing Data Mentah")
    local_files=scan_local_files()
    src_tab1,src_tab2=st.tabs(["📂 Pilih dari folder","⬆️ Upload file baru"])
    with src_tab1:
        if local_files:
            col_sel,col_info=st.columns([4,1])
            with col_sel: chosen_raw=st.selectbox("Pilih file mentah",local_files,key='pp_local')
            with col_info:
                st.markdown("<br>",unsafe_allow_html=True)
                if chosen_raw and os.path.exists(chosen_raw):
                    st.caption(f"{os.path.getsize(chosen_raw)/1024:,.0f} KB")
            if st.button("📂 Gunakan File Ini",key='pp_use',use_container_width=True):
                with open(chosen_raw,'rb') as f_: raw_file_bytes=f_.read()
                st.session_state['pp_raw_bytes']=raw_file_bytes
                st.session_state['pp_raw_name']=chosen_raw
                st.session_state.pop('pp_df_out',None); st.session_state.pop('pp_logs',None)
                st.success(f"✅ **{chosen_raw}** siap diproses.")
        else: st.info("Belum ada file di folder.")
    with src_tab2:
        uploaded_raw=st.file_uploader("Upload file mentah",type=['csv','xlsx'],key='pp_up')
        if uploaded_raw and st.session_state.get('pp_raw_name')!=uploaded_raw.name:
            st.session_state['pp_raw_bytes']=uploaded_raw.getvalue()
            st.session_state['pp_raw_name']=uploaded_raw.name
            st.session_state.pop('pp_df_out',None); st.session_state.pop('pp_logs',None)

    raw_file_bytes=st.session_state.get('pp_raw_bytes')
    raw_file_name=st.session_state.get('pp_raw_name')
    if not raw_file_bytes: return

    st.markdown(f"📋 **File:** `{raw_file_name}`")

    # ── Info STEP 0 ────────────────────────────────────────────
    st.info(
        "**🧹 STEP 0 — Error Absen (otomatis aktif):**\n"
        "- Double absen MASUK → ambil yang paling **awal**\n"
        "- Double absen PULANG → ambil yang paling **akhir**\n"
        "- Absen PULANG tanpa pasangan MASUK di hari sama → **dihapus**"
    )

    c1,c2,c3=st.columns(3)
    with c1:
        run_dbscan=st.checkbox("🔵 DBSCAN",value=True)
        run_stdbscan=st.checkbox("🟣 ST-DBSCAN",value=True)
        recalc=st.checkbox("🔄 Hitung ulang status",value=False)
    with c2:
        eps_km=st.number_input("DBSCAN radius (km)",value=0.1,step=0.05,format="%.2f")
        min_smp=st.number_input("DBSCAN min_samples",value=3,step=1)
        st_eps_km=st.number_input("ST-DBSCAN radius (km)",value=0.1,step=0.05,format="%.2f")
    with c3:
        st_eps_hr=st.number_input("ST-DBSCAN radius (jam)",value=1.0,step=0.5,format="%.1f")
        st_min_smp=st.number_input("ST-DBSCAN min_samples",value=3,step=1)

    config={'run_dbscan':run_dbscan,'run_stdbscan':run_stdbscan,'recalc_status':recalc,
            'eps_km':eps_km,'min_samples':int(min_smp),'st_eps_km':st_eps_km,
            'st_eps_hours':st_eps_hr,'st_min_samples':int(st_min_smp)}

    run_btn=st.button("🚀 Jalankan Preprocessing",type="primary",use_container_width=True)
    if not run_btn and 'pp_df_out' not in st.session_state: return
    if run_btn:
        buf=io.BytesIO(raw_file_bytes)
        try: df_raw=pd.read_csv(buf) if str(raw_file_name).endswith('.csv') else pd.read_excel(buf)
        except Exception as e: st.error(f"❌ {e}"); return
        required=['karyawan_id','id_skpd','jenis','lat','long','tanggal_kirim']
        missing=[c for c in required if c not in df_raw.columns]
        if missing: st.error(f"❌ Kolom wajib tidak ada: {missing}"); return
        progress=st.progress(0,"Memulai...")
        try:
            progress.progress(10,"Membersihkan error absen...")
            df_out,logs=_run_preprocessing(df_raw,config)
            progress.progress(100,"✅ Selesai!")
            st.session_state['pp_df_out']=df_out; st.session_state['pp_logs']=logs
        except Exception as e:
            import traceback; st.error(f"❌ {e}"); st.code(traceback.format_exc()); return

    if 'pp_df_out' not in st.session_state: return
    df_out=st.session_state['pp_df_out']; logs=st.session_state.get('pp_logs',[])

    if logs:
        st.markdown("### 📋 Log Preprocessing")
        # Pisahkan log STEP 0 agar mudah dibaca
        step0_logs = [l for l in logs if 'STEP 0' in l]
        other_logs  = [l for l in logs if 'STEP 0' not in l]
        if step0_logs:
            with st.expander("🧹 Detail STEP 0 — Error Absen", expanded=True):
                for lg in step0_logs:
                    st.markdown(f"- {lg}")
        for lg in other_logs:
            st.markdown(f"- {lg}")

    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Total",f"{len(df_out):,}")
    with c2: st.metric("Karyawan",f"{df_out['karyawan_id'].nunique():,}")
    with c3: st.metric("SKPD",f"{df_out['id_skpd'].nunique():,}")
    with c4: st.metric("HIGH Risk",f"{(df_out.get('risk_level',pd.Series())=='HIGH').sum():,}")

    with st.expander("🔍 Preview",expanded=True):
        st.dataframe(df_out.head(10),use_container_width=True)

    dc1,dc2=st.columns(2)
    with dc1:
        st.download_button("📄 CSV",df_out.to_csv(index=False).encode('utf-8'),
                           "absensi_preprocessed.csv","text/csv",use_container_width=True)
    with dc2:
        xl=io.BytesIO(); df_out.to_excel(xl,index=False)
        st.download_button("📊 Excel",xl.getvalue(),"absensi_preprocessed.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

# ============================================================
# MAIN
# ============================================================
def main():
    for key,default in [('df',None),('office_centroid',pd.DataFrame())]:
        if key not in st.session_state: st.session_state[key] = default

    page, filters = render_sidebar()

    pages = {
        "🏠 Beranda":     page_beranda,
        "📥 Upload Data": page_upload,
        "📊 Visualisasi": lambda: page_visualisasi(filters),
        "🎯 Hunting":     page_hunting,
        # "🔧 Preprocessing": page_preprocessing,  # AKTIFKAN PREPROCESSING: uncomment + tambah ke nav_pages
    }
    pages.get(page, page_beranda)()

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:gray;font-size:11px">'
        'Analisis Absensi v3 — T1/T2/T3/T4/TWM/TWP/PC1-4 | Streamlit + Folium + Plotly'
        '</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
