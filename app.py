"""
Visualisasi & Deteksi Anomali Absensi
Streamlit App - Integrated Preprocessing + Visualization + Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN
import pydeck as pdk
import io
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Deteksi Anomali Absensi",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:bold;color:#1f77b4;text-align:center;padding:1rem 0 0.2rem 0}
.sub-header{text-align:center;color:#666;font-size:.95rem;margin-bottom:1.5rem}
.step-box{background:#f8f9fa;border-left:4px solid #1f77b4;padding:.8rem 1rem;border-radius:0 8px 8px 0;margin-bottom:.8rem}
.step-title{font-weight:bold;color:#1f77b4;font-size:1rem}
.step-desc{color:#444;font-size:.88rem;margin-top:.2rem}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PREPROCESSING ENGINE
# ============================================================

def haversine_scalar(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    dlat  = np.radians(lat2 - lat1)
    dlon  = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def run_full_pipeline(df_raw: pd.DataFrame, params: dict):
    """
    Full preprocessing pipeline.
    Returns (df_result, log_dict)
    """
    log = {}
    df = df_raw.copy()

    # ── STEP 1: DATA SELECTION ──────────────────────────────
    cols_needed  = ['karyawan_id','lat','long','tanggal_kirim','jenis','id_skpd','catatan','status_lokasi']
    cols_optional = ['jarak','created_at','status_presensi','approver_status']
    cols_use = [c for c in cols_needed if c in df.columns]
    cols_use += [c for c in cols_optional if c in df.columns]
    df = df[cols_use].copy()
    log['step1_cols']  = cols_use
    log['step1_rows']  = len(df)

    # ── STEP 2: DATA PROFILING ──────────────────────────────
    log['profile_null']        = df.isnull().sum().to_dict()
    log['profile_rows_before'] = len(df)
    if 'lat' in df.columns and 'long' in df.columns:
        log['profile_coord_error'] = int(df[
            (~df['lat'].between(-90,90)) | (~df['long'].between(-180,180))
        ].shape[0])
    dup_cols = [c for c in ['karyawan_id','tanggal_kirim','jenis'] if c in df.columns]
    log['profile_duplicates'] = int(df.duplicated(subset=dup_cols).sum())

    # ── STEP 3: DATA CLEANING ───────────────────────────────
    df = df.dropna(subset=['lat','long'])
    if 'jarak' in df.columns:
        df['jarak'] = pd.to_numeric(df['jarak'], errors='coerce')
        df = df.dropna(subset=['jarak'])
    df = df[(df['lat'].between(-90,90)) & (df['long'].between(-180,180))]
    df = df[~((df['lat']==0) & (df['long']==0))]
    df = df.drop_duplicates(subset=dup_cols)
    log['step3_rows_after'] = len(df)
    log['step3_dropped']    = log['profile_rows_before'] - len(df)

    # ── STEP 4: TIME TRANSFORMATION ─────────────────────────
    df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'], errors='coerce')
    df = df.dropna(subset=['tanggal_kirim'])
    df['timestamp_num'] = df['tanggal_kirim'].astype('int64') // 10**9
    df['jam']           = df['tanggal_kirim'].dt.hour
    df['menit']         = df['tanggal_kirim'].dt.minute
    df['jam_desimal']   = df['jam'] + df['menit'] / 60.0
    df['tanggal']       = df['tanggal_kirim'].dt.date
    df['weekday']       = df['tanggal_kirim'].dt.weekday
    log['step4_rows'] = len(df)

    # ── STEP 5: COORDINATE TRANSFORMATION ───────────────────
    df['lat_rad']  = np.radians(df['lat'])
    df['long_rad'] = np.radians(df['long'])

    # ── STEP 6: NORMALISASI JENIS ────────────────────────────
    df['jenis'] = df['jenis'].astype(str).str.strip().str.upper()
    df_masuk  = df[df['jenis']=='M'].copy()
    df_pulang = df[df['jenis']=='P'].copy()
    log['step6_masuk']  = len(df_masuk)
    log['step6_pulang'] = len(df_pulang)

    # ── STEP 7: DBSCAN SPATIAL ──────────────────────────────
    eps_km      = params.get('eps_km', 0.2)
    min_samples = params.get('min_samples', 20)
    eps_rad     = eps_km / 6371.0

    def run_dbscan(subset, label_col):
        if len(subset) < min_samples:
            subset[label_col] = -1
            return subset
        coords = subset[['lat_rad','long_rad']].to_numpy()
        labels = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine').fit_predict(coords)
        subset[label_col] = labels
        return subset

    df_masuk  = run_dbscan(df_masuk,  'cluster_masuk')
    df_pulang = run_dbscan(df_pulang, 'cluster_pulang')

    log['step7_clusters_masuk']  = int(df_masuk['cluster_masuk'].nunique()  - (1 if -1 in df_masuk['cluster_masuk'].values  else 0))
    log['step7_noise_masuk']     = int((df_masuk['cluster_masuk']  == -1).sum())
    log['step7_clusters_pulang'] = int(df_pulang['cluster_pulang'].nunique() - (1 if -1 in df_pulang['cluster_pulang'].values else 0))
    log['step7_noise_pulang']    = int((df_pulang['cluster_pulang'] == -1).sum())

    # ── STEP 8: ESTIMASI KANTOR PER SKPD ────────────────────
    df_clustered = df_masuk[df_masuk['cluster_masuk'] != -1].copy()
    if len(df_clustered) > 0:
        cluster_size = (
            df_clustered.groupby(['id_skpd','cluster_masuk'])
            .size().reset_index(name='n')
        )
        idx = cluster_size.groupby('id_skpd')['n'].idxmax()
        main_cluster = cluster_size.loc[idx]
        df_office = df_clustered.merge(
            main_cluster[['id_skpd','cluster_masuk']],
            on=['id_skpd','cluster_masuk'], how='inner'
        )
        office_centroid = (
            df_office.groupby('id_skpd')[['lat','long']].mean()
            .reset_index()
            .rename(columns={'lat':'office_lat','long':'office_long'})
        )
    else:
        # Fallback: median per SKPD
        office_centroid = (
            df_masuk.groupby('id_skpd')[['lat','long']].median()
            .reset_index()
            .rename(columns={'lat':'office_lat','long':'office_long'})
        )
    log['step8_offices'] = len(office_centroid)

    # ── STEP 9: MERGE & HAVERSINE DISTANCE ──────────────────
    df_model = df.merge(office_centroid, on='id_skpd', how='left')
    df_model['dist_km'] = haversine_vectorized(
        df_model['lat'].values, df_model['long'].values,
        df_model['office_lat'].fillna(df_model['lat']).values,
        df_model['office_long'].fillna(df_model['long']).values
    )
    df_model['outside_300m'] = (df_model['dist_km'] > 0.3).astype(int)
    df_model['very_far']     = (df_model['dist_km'] > 5.0).astype(int)
    df_model['extreme_far']  = (df_model['dist_km'] > 50.0).astype(int)

    # ── STEP 10: VALIDATION LOGIC ───────────────────────────
    df_model['no_note']          = df_model['catatan'].isna().astype(int)
    df_model['far_no_note']      = ((df_model['outside_300m']==1) & (df_model['no_note']==1)).astype(int)
    df_model['far_with_note']    = ((df_model['outside_300m']==1) & (df_model['no_note']==0)).astype(int)
    if 'status_lokasi' in df_model.columns:
        df_model['near_but_status0'] = ((df_model['outside_300m']==0) & (df_model['status_lokasi']==0)).astype(int)
        df_model['far_but_status1']  = ((df_model['outside_300m']==1) & (df_model['status_lokasi']==1)).astype(int)
    else:
        df_model['near_but_status0'] = 0
        df_model['far_but_status1']  = 0

    # ── STEP 11: MERGE CLUSTER MASUK & PULANG ───────────────
    cluster_masuk_map  = df_masuk[['karyawan_id','tanggal_kirim','cluster_masuk']].copy()
    cluster_pulang_map = df_pulang[['karyawan_id','tanggal_kirim','cluster_pulang']].copy()
    df_model = df_model.merge(cluster_masuk_map,  on=['karyawan_id','tanggal_kirim'], how='left')
    df_model = df_model.merge(cluster_pulang_map, on=['karyawan_id','tanggal_kirim'], how='left')

    df_model['is_noise_masuk']  = ((df_model['jenis']=='M') & (df_model['cluster_masuk']==-1)).astype(int)
    df_model['is_noise_pulang'] = ((df_model['jenis']=='P') & (df_model['cluster_pulang']==-1)).astype(int)

    size_masuk  = df_masuk['cluster_masuk'].value_counts()
    size_pulang = df_pulang['cluster_pulang'].value_counts()
    df_model['cluster_size_masuk']  = df_model['cluster_masuk'].map(size_masuk).fillna(0).astype(int)
    df_model['cluster_size_pulang'] = df_model['cluster_pulang'].map(size_pulang).fillna(0).astype(int)

    # ── STEP 12: ST-DBSCAN ──────────────────────────────────
    eps_space_km  = params.get('st_eps_km', 0.2)
    eps_space_rad = eps_space_km / 6371.0
    st_min        = params.get('st_min_samples', 15)
    eps_time_h    = params.get('st_eps_hours', 0.5)
    time_scale    = eps_space_rad / eps_time_h

    def run_st_dbscan(subset, label_col):
        if len(subset) < st_min:
            subset[label_col] = -1
            return subset
        space = subset[['lat_rad','long_rad']].to_numpy()
        time  = subset[['jam_desimal']].to_numpy() * time_scale
        X     = np.hstack([space, time])
        labels = DBSCAN(eps=eps_space_rad, min_samples=st_min, metric='euclidean').fit_predict(X)
        subset[label_col] = labels
        return subset

    df_st_masuk  = df_model[df_model['jenis']=='M'].copy()
    df_st_pulang = df_model[df_model['jenis']=='P'].copy()
    df_st_masuk  = run_st_dbscan(df_st_masuk,  'st_cluster_masuk')
    df_st_pulang = run_st_dbscan(df_st_pulang, 'st_cluster_pulang')

    st_masuk_map  = df_st_masuk[['karyawan_id','tanggal_kirim','st_cluster_masuk']].copy()
    st_pulang_map = df_st_pulang[['karyawan_id','tanggal_kirim','st_cluster_pulang']].copy()
    df_model = df_model.merge(st_masuk_map,  on=['karyawan_id','tanggal_kirim'], how='left')
    df_model = df_model.merge(st_pulang_map, on=['karyawan_id','tanggal_kirim'], how='left')

    df_model['is_st_noise_masuk']  = ((df_model['jenis']=='M') & (df_model['st_cluster_masuk']==-1)).astype(int)
    df_model['is_st_noise_pulang'] = ((df_model['jenis']=='P') & (df_model['st_cluster_pulang']==-1)).astype(int)

    log['step12_st_noise_masuk']  = int(df_model['is_st_noise_masuk'].sum())
    log['step12_st_noise_pulang'] = int(df_model['is_st_noise_pulang'].sum())

    # ── STEP 13: ANOMALY SCORING ────────────────────────────
    df_model['anomaly_score'] = (
        df_model['outside_300m']      * 25 +
        df_model['far_no_note']       * 35 +
        df_model['very_far']          * 30 +
        df_model['extreme_far']       * 50 +
        df_model['near_but_status0']  *  5 +
        df_model['is_noise_masuk']    * 15 +
        df_model['is_noise_pulang']   * 10 +
        df_model['is_st_noise_masuk'] * 10
    )

    def risk_level(s):
        if s >= 70:   return 'HIGH'
        elif s >= 30: return 'MED'
        else:         return 'LOW'

    df_model['risk_level'] = df_model['anomaly_score'].apply(risk_level)

    def system_action(risk):
        if risk == 'LOW':  return 'AUTO APPROVE'
        elif risk == 'MED': return 'HOLD (Perlu Review)'
        else:               return 'TEMP REJECT + NOTIF APPROVER'

    df_model['system_action'] = df_model['risk_level'].apply(system_action)

    log['step13_high'] = int((df_model['risk_level']=='HIGH').sum())
    log['step13_med']  = int((df_model['risk_level']=='MED').sum())
    log['step13_low']  = int((df_model['risk_level']=='LOW').sum())
    log['step13_total'] = len(df_model)

    # Simpan office_centroid di log untuk peta
    log['office_centroid'] = office_centroid

    return df_model, log


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_color_folium(risk):
    return {'HIGH':'red','MED':'orange','LOW':'green'}.get(risk,'blue')

def get_risk_color_hex(risk):
    return {'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'}.get(risk,'#3498db')

def build_popup(row):
    risk  = row.get('risk_level','N/A')
    color = get_risk_color_hex(risk)
    dist  = row.get('dist_km', 0)
    score = row.get('anomaly_score', 0)
    action = row.get('system_action', '-')
    catatan = row.get('catatan', '-')
    if pd.isna(catatan): catatan = '-'
    return f"""
    <div style='font-family:Arial;font-size:12px;min-width:230px'>
        <h4 style='margin:0 0 8px 0;color:#2c3e50'>📋 Detail Absensi</h4>
        <table style='width:100%;border-collapse:collapse'>
            <tr><td><b>Karyawan</b></td><td>{row.get('karyawan_id','N/A')}</td></tr>
            <tr><td><b>SKPD</b></td><td>{row.get('id_skpd','N/A')}</td></tr>
            <tr><td><b>Jenis</b></td><td>{'🟢 Masuk' if row.get('jenis')=='M' else '🔴 Pulang'}</td></tr>
            <tr><td><b>Waktu</b></td><td>{str(row.get('tanggal_kirim','N/A'))[:16]}</td></tr>
            <tr><td><b>Jarak ke kantor</b></td><td>{dist:.3f} km</td></tr>
            <tr><td><b>Anomaly Score</b></td><td>{score}</td></tr>
            <tr><td><b>Risk Level</b></td><td><span style='color:{color};font-weight:bold'>{risk}</span></td></tr>
            <tr><td><b>Aksi Sistem</b></td><td>{action}</td></tr>
            <tr><td><b>Catatan</b></td><td>{catatan}</td></tr>
        </table>
    </div>"""


def create_folium_map(df, map_type='marker', office_centroid=None):
    center_lat  = df['lat'].median()
    center_long = df['long'].median()
    m = folium.Map(location=[center_lat, center_long], zoom_start=13, tiles='OpenStreetMap')
    folium.TileLayer('CartoDB positron', name='CartoDB Light').add_to(m)

    if map_type == 'heatmap':
        heat_data = [[r['lat'], r['long'], r.get('anomaly_score',1)+1] for _, r in df.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=14).add_to(m)
    elif map_type == 'cluster':
        mc = MarkerCluster(name='Absensi').add_to(m)
        for _, row in df.iterrows():
            color = get_risk_color_folium(row.get('risk_level','LOW'))
            folium.CircleMarker(
                location=[row['lat'], row['long']], radius=7,
                color=color, fill=True, fill_color=color, fill_opacity=0.75,
                popup=folium.Popup(build_popup(row), max_width=280)
            ).add_to(mc)
    else:
        for risk in ['HIGH','MED','LOW']:
            fg = folium.FeatureGroup(name=f'Risk {risk}')
            color = get_risk_color_folium(risk)
            for _, row in df[df['risk_level']==risk].iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=8 if risk=='HIGH' else 6,
                    color=color, fill=True, fill_color=color, fill_opacity=0.7,
                    popup=folium.Popup(build_popup(row), max_width=280)
                ).add_to(fg)
            fg.add_to(m)

    # Marker kantor
    if office_centroid is not None and len(office_centroid) > 0:
        fg_office = folium.FeatureGroup(name='Kantor SKPD')
        for _, row in office_centroid.iterrows():
            if pd.notna(row['office_lat']):
                folium.Marker(
                    location=[row['office_lat'], row['office_long']],
                    popup=f"<b>Kantor SKPD {row['id_skpd']}</b>",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa')
                ).add_to(fg_office)
                folium.Circle(
                    location=[row['office_lat'], row['office_long']],
                    radius=300, color='blue', fill=False, weight=2, dash_array='5'
                ).add_to(fg_office)
        fg_office.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def create_pydeck_map(df):
    DEFAULT_COLOR = [52, 152, 219, 200]
    color_map = {'HIGH':[231,76,60,200],'MED':[243,156,18,200],'LOW':[39,174,96,200]}
    df_map = df.copy()
    df_map['color']  = df_map['risk_level'].apply(lambda x: color_map.get(x, DEFAULT_COLOR))
    df_map['radius'] = df_map['anomaly_score'].fillna(0) * 10 + 20

    scatter = pdk.Layer(
        'ScatterplotLayer', data=df_map,
        get_position='[long, lat]', get_color='color', get_radius='radius',
        pickable=True, opacity=0.8, stroked=True, filled=True,
        radius_scale=6, radius_min_pixels=4, radius_max_pixels=30
    )
    view = pdk.ViewState(
        latitude=df['lat'].median(), longitude=df['long'].median(),
        zoom=12, pitch=40, bearing=0
    )
    return pdk.Deck(
        layers=[scatter], initial_view_state=view,
        tooltip={
            'html': '<b>Karyawan:</b> {karyawan_id}<br><b>Risk:</b> {risk_level}<br>'
                    '<b>Score:</b> {anomaly_score}<br><b>Jarak:</b> {dist_km} km',
            'style': {'backgroundColor':'steelblue','color':'white'}
        }
    )


def apply_filters(df, selected_skpd, risk_options, jenis_options, date_range, dist_range):
    f = df.copy()
    if selected_skpd != 'Semua':
        f = f[f['id_skpd'] == selected_skpd]
    if risk_options:
        f = f[f['risk_level'].isin(risk_options)]
    if jenis_options:
        f = f[f['jenis'].isin(jenis_options)]
    if date_range and len(date_range) == 2 and 'tanggal_kirim' in f.columns:
        f = f[(f['tanggal_kirim'].dt.date >= date_range[0]) & (f['tanggal_kirim'].dt.date <= date_range[1])]
    if 'dist_km' in f.columns:
        f = f[(f['dist_km'] >= dist_range[0]) & (f['dist_km'] <= dist_range[1])]
    return f


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    st.sidebar.markdown("## 🗺️ Deteksi Anomali Absensi")
    st.sidebar.markdown("---")

    # ── NAVIGASI ──────────────────────────────────────────
    page = st.sidebar.radio(
        "📌 Navigasi",
        options=["🏠 Beranda", "📥 Upload & Preprocessing", "📊 Visualisasi", "🔮 Prediksi"],
        index=0
    )

    st.sidebar.markdown("---")

    # ── UPLOAD FILE ───────────────────────────────────────
    st.sidebar.markdown("### 📂 Upload Data")
    uploaded = st.sidebar.file_uploader(
        "Upload file absensi (CSV/Excel)",
        type=['csv','xlsx'],
        help="Upload file absen_pegawai.xlsx atau hasil preprocessing"
    )

    # ── PARAMETER PREPROCESSING ───────────────────────────
    with st.sidebar.expander("⚙️ Parameter Preprocessing", expanded=False):
        eps_km      = st.slider("DBSCAN Radius (km)",      0.05, 1.0, 0.2, 0.05)
        min_samples = st.slider("DBSCAN Min Samples",       5, 50, 20, 5)
        st_eps_km   = st.slider("ST-DBSCAN Radius (km)",   0.05, 1.0, 0.2, 0.05)
        st_min      = st.slider("ST-DBSCAN Min Samples",    5, 50, 15, 5)
        st_eps_h    = st.slider("ST-DBSCAN Waktu (jam)",   0.1, 2.0, 0.5, 0.1)

    params = {
        'eps_km': eps_km, 'min_samples': min_samples,
        'st_eps_km': st_eps_km, 'st_min_samples': st_min, 'st_eps_hours': st_eps_h
    }

    # ── FILTER (hanya tampil di halaman Visualisasi) ───────
    filters = {}
    if 'df_result' in st.session_state and st.session_state.df_result is not None:
        df = st.session_state.df_result
        with st.sidebar.expander("🔍 Filter Data", expanded=True):
            skpd_list = ['Semua'] + sorted(df['id_skpd'].unique().tolist())
            filters['skpd']   = st.selectbox("SKPD", skpd_list)
            filters['risk']   = st.multiselect("Risk Level", ['HIGH','MED','LOW'], default=['HIGH','MED','LOW'])
            filters['jenis']  = st.multiselect("Jenis", ['M','P'], default=['M','P'],
                                               format_func=lambda x: 'Masuk' if x=='M' else 'Pulang')
            if 'tanggal_kirim' in df.columns:
                min_d = df['tanggal_kirim'].min().date()
                max_d = df['tanggal_kirim'].max().date()
                filters['date'] = st.date_input("Rentang Tanggal", value=(min_d, max_d),
                                                min_value=min_d, max_value=max_d)
            else:
                filters['date'] = None
            max_dist = float(df['dist_km'].max()) if 'dist_km' in df.columns else 100.0
            filters['dist'] = st.slider("Jarak ke Kantor (km)", 0.0, min(max_dist,100.0),
                                        (0.0, min(max_dist,100.0)), 0.1)

        with st.sidebar.expander("🗺️ Pengaturan Peta", expanded=False):
            filters['map_type']  = st.radio("Tipe Peta", ['marker','cluster','heatmap'],
                                            format_func=lambda x: {'marker':'📍 Marker','cluster':'🔵 Cluster','heatmap':'🔥 Heatmap'}[x])
            filters['use_pydeck'] = st.checkbox("Peta 3D (PyDeck)", value=False)

    return page, uploaded, params, filters


# ============================================================
# PAGE: BERANDA
# ============================================================

def page_beranda():
    st.markdown('<div class="main-header">🗺️ Deteksi Anomali Absensi Pegawai</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistem deteksi anomali lokasi absensi menggunakan DBSCAN, ST-DBSCAN, dan Haversine Distance</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 📥 Step 1\n**Upload & Preprocessing**\nUpload file absensi Anda, sistem akan otomatis melakukan preprocessing lengkap.")
    with col2:
        st.success("### 📊 Step 2\n**Visualisasi**\nLihat peta interaktif, analisis temporal, distribusi jarak, dan statistik anomali.")
    with col3:
        st.warning("### 🔮 Step 3\n**Prediksi**\nUpload data baru untuk diprediksi apakah termasuk anomali atau tidak.")

    st.markdown("---")
    st.markdown("### 🔄 Alur Pipeline")

    pipeline_steps = [
        ("1️⃣ Data Selection",       "Pilih kolom yang relevan: karyawan_id, lat, long, tanggal_kirim, jenis, id_skpd, catatan, status_lokasi"),
        ("2️⃣ Data Profiling",        "Cek null values, koordinat error, duplikat absensi — laporan kesehatan data"),
        ("3️⃣ Data Cleaning",         "Drop koordinat kosong/invalid, buang (0,0), hapus duplikat, validasi jarak numerik"),
        ("4️⃣ Time Transformation",   "Ekstrak jam, menit, jam_desimal, weekday, timestamp_num dari tanggal_kirim"),
        ("5️⃣ Coordinate Transform",  "Konversi lat/long ke radian untuk DBSCAN metric haversine"),
        ("6️⃣ Split Masuk/Pulang",    "Pisahkan data M (Masuk) dan P (Pulang) untuk clustering terpisah"),
        ("7️⃣ DBSCAN Spatial",        "Clustering lokasi per jenis absensi — temukan hotspot dan noise"),
        ("8️⃣ Estimasi Kantor",       "Ambil cluster terbesar per SKPD → hitung centroid sebagai lokasi kantor"),
        ("9️⃣ Haversine Distance",    "Hitung jarak setiap absensi ke kantor SKPD-nya (km)"),
        ("🔟 Validation Logic",      "Flag: outside_300m, far_no_note, far_with_note, near_but_status0"),
        ("1️⃣1️⃣ ST-DBSCAN",          "Clustering spatio-temporal — deteksi pola tidak konsisten waktu+lokasi"),
        ("1️⃣2️⃣ Anomaly Scoring",    "Hitung skor risiko dari kombinasi semua sinyal anomali"),
        ("1️⃣3️⃣ Risk Level",         "Klasifikasi: LOW (0-29) | MED (30-69) | HIGH (≥70)"),
    ]

    for title, desc in pipeline_steps:
        st.markdown(f"""
        <div class="step-box">
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Anomaly Scoring Formula")
    score_data = {
        'Sinyal': ['outside_300m (>300m)', 'far_no_note (jauh+tanpa alasan)', 'very_far (>5km)',
                   'extreme_far (>50km)', 'near_but_status0 (mismatch)', 'is_noise_masuk (DBSCAN noise)',
                   'is_noise_pulang', 'is_st_noise_masuk (ST-DBSCAN noise)'],
        'Bobot': [25, 35, 30, 50, 5, 15, 10, 10],
        'Keterangan': [
            'Di luar radius 300m dari kantor',
            'Jauh dari kantor + tidak ada catatan alasan',
            'Sangat jauh (>5km)',
            'Ekstrem jauh >50km — indikasi fake GPS',
            'Dekat tapi sistem lama bilang di luar',
            'Tidak masuk cluster lokasi manapun',
            'Tidak masuk cluster pulang manapun',
            'Tidak konsisten secara ruang-waktu'
        ]
    }
    st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

    st.markdown("""
    | Score | Risk Level | Aksi Sistem |
    |---|---|---|
    | 0 – 29 | 🟢 LOW | AUTO APPROVE |
    | 30 – 69 | 🟡 MED | HOLD (Perlu Review) |
    | ≥ 70 | 🔴 HIGH | TEMP REJECT + NOTIF APPROVER |
    """)


# ============================================================
# PAGE: UPLOAD & PREPROCESSING
# ============================================================

def page_preprocessing(uploaded, params):
    st.markdown("## 📥 Upload & Preprocessing")

    if uploaded is None:
        # Coba load dari file lokal
        if os.path.exists('absen_pegawai.xlsx'):
            st.info("📂 Menggunakan file `absen_pegawai.xlsx` yang ditemukan di folder.")
            df_raw = pd.read_excel('absen_pegawai.xlsx')
        elif os.path.exists('dataset_absensi_final2.xlsx'):
            st.info("📂 Menggunakan file `dataset_absensi_final2.xlsx` yang ditemukan di folder.")
            df_raw = pd.read_excel('dataset_absensi_final2.xlsx')
        else:
            st.warning("⬆️ Silakan upload file absensi melalui sidebar kiri (CSV atau Excel).")
            st.markdown("""
            **Format kolom yang dibutuhkan:**
            | Kolom | Tipe | Keterangan |
            |---|---|---|
            | karyawan_id | integer | ID pegawai |
            | id_skpd | integer | ID kantor/SKPD |
            | lat | float | Latitude absensi |
            | long | float | Longitude absensi |
            | tanggal_kirim | datetime | Waktu absensi |
            | jenis | string | M=Masuk, P=Pulang |
            | status_lokasi | integer | 1=dalam, 0=luar |
            | catatan | string | Alasan tugas luar |
            """)
            return
    else:
        if uploaded.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)

    st.success(f"✅ File berhasil dimuat: **{len(df_raw):,} baris**, **{len(df_raw.columns)} kolom**")

    # ── PROFILING AWAL ─────────────────────────────────────
    with st.expander("🔍 Data Profiling (Sebelum Cleaning)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Baris", f"{len(df_raw):,}")
        with col2:
            st.metric("Total Kolom", len(df_raw.columns))
        with col3:
            null_total = df_raw.isnull().sum().sum()
            st.metric("Total Null Values", f"{null_total:,}")

        st.markdown("**Kolom & Tipe Data:**")
        dtype_df = pd.DataFrame({
            'Kolom': df_raw.dtypes.index,
            'Tipe': df_raw.dtypes.values.astype(str),
            'Null': df_raw.isnull().sum().values,
            'Null %': (df_raw.isnull().sum().values / len(df_raw) * 100).round(2)
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        st.markdown("**Sample Data (5 baris pertama):**")
        st.dataframe(df_raw.head(), use_container_width=True)

    # ── JALANKAN PIPELINE ──────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Jalankan Preprocessing Pipeline")
    st.markdown("Klik tombol di bawah untuk menjalankan semua tahap preprocessing secara otomatis.")

    if st.button("🚀 Jalankan Preprocessing", type="primary", use_container_width=True):
        with st.spinner("⏳ Memproses data... Mohon tunggu..."):
            try:
                df_result, log = run_full_pipeline(df_raw, params)
                st.session_state['df_result'] = df_result
                st.session_state['log']       = log
                st.session_state['df_raw']    = df_raw
                st.success("✅ Preprocessing selesai!")
            except Exception as e:
                st.error(f"❌ Error saat preprocessing: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ── TAMPILKAN HASIL ────────────────────────────────────
    if 'df_result' in st.session_state and st.session_state.df_result is not None:
        df_result = st.session_state.df_result
        log       = st.session_state.log

        st.markdown("---")
        st.markdown("### 📋 Hasil Preprocessing — Step by Step")

        # Step 1
        with st.expander("✅ Step 1: Data Selection", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">Memilih kolom-kolom yang relevan untuk analisis anomali. Kolom tidak relevan dibuang untuk efisiensi.</div>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Kolom yang digunakan:** {log.get('step1_cols', [])}")
            st.write(f"**Jumlah baris:** {log.get('step1_rows', 0):,}")

        # Step 2
        with st.expander("✅ Step 2: Data Profiling", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">Menganalisis kualitas data: null values, koordinat di luar range bumi, dan duplikat absensi.</div>
            </div>
            """, unsafe_allow_html=True)
            null_data = {k: v for k, v in log.get('profile_null', {}).items() if v > 0}
            if null_data:
                st.write("**Null Values per Kolom:**")
                st.dataframe(pd.DataFrame(list(null_data.items()), columns=['Kolom','Null Count']),
                             use_container_width=True, hide_index=True)
            else:
                st.success("Tidak ada null values pada kolom utama.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Koordinat Error", log.get('profile_coord_error', 0))
            with col2:
                st.metric("Duplikat Absensi", log.get('profile_duplicates', 0))

        # Step 3
        with st.expander("✅ Step 3: Data Cleaning", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    1. Drop baris dengan koordinat kosong (lat/long null)<br>
                    2. Validasi jarak numerik (kolom jarak jika ada)<br>
                    3. Filter koordinat di luar range bumi (-90 s/d 90, -180 s/d 180)<br>
                    4. Buang koordinat (0,0) — GPS tidak valid<br>
                    5. Hapus duplikat absensi (karyawan_id + tanggal_kirim + jenis)
                </div>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Baris Sebelum", f"{log.get('profile_rows_before',0):,}")
            with col2:
                st.metric("Baris Setelah", f"{log.get('step3_rows_after',0):,}")
            with col3:
                st.metric("Baris Dibuang", f"{log.get('step3_dropped',0):,}",
                          delta=f"-{log.get('step3_dropped',0)}", delta_color="inverse")

        # Step 4
        with st.expander("✅ Step 4: Time Transformation", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    Mengekstrak fitur waktu dari kolom tanggal_kirim:<br>
                    • <b>timestamp_num</b>: detik sejak epoch (untuk analisa lintas hari)<br>
                    • <b>jam</b>: jam absensi (0-23)<br>
                    • <b>menit</b>: menit absensi (0-59)<br>
                    • <b>jam_desimal</b>: jam + menit/60 (contoh: 7:30 → 7.5) — dipakai ST-DBSCAN<br>
                    • <b>weekday</b>: hari dalam minggu (0=Senin, 6=Minggu)
                </div>
            </div>
            """, unsafe_allow_html=True)
            if 'jam_desimal' in df_result.columns:
                fig = px.histogram(df_result, x='jam_desimal', nbins=48,
                                   title='Distribusi Jam Absensi',
                                   labels={'jam_desimal':'Jam (desimal)','count':'Jumlah'})
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)

        # Step 5
        with st.expander("✅ Step 5: Coordinate Transformation", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    Konversi lat/long dari derajat ke radian menggunakan np.radians().<br>
                    <b>Mengapa?</b> DBSCAN dengan metric='haversine' di scikit-learn membutuhkan input dalam radian,
                    bukan derajat. Tanpa konversi ini, jarak yang dihitung akan salah.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if 'lat_rad' in df_result.columns:
                st.dataframe(df_result[['lat','long','lat_rad','long_rad']].head(5),
                             use_container_width=True)

        # Step 6
        with st.expander("✅ Step 6: Split Masuk / Pulang", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    Memisahkan data menjadi dua subset: M (Masuk) dan P (Pulang).<br>
                    <b>Mengapa?</b> Pola lokasi masuk dan pulang bisa berbeda. Clustering terpisah
                    menghasilkan hotspot yang lebih akurat untuk masing-masing jenis absensi.
                </div>
            </div>
            """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data MASUK (M)", f"{log.get('step6_masuk',0):,}")
            with col2:
                st.metric("Data PULANG (P)", f"{log.get('step6_pulang',0):,}")

        # Step 7
        with st.expander("✅ Step 7: DBSCAN Spatial Clustering", expanded=False):
            st.markdown(f"""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    DBSCAN (Density-Based Spatial Clustering) mengelompokkan titik-titik absensi
                    berdasarkan kepadatan lokasi.<br><br>
                    <b>Parameter yang digunakan:</b><br>
                    • eps = {params['eps_km']} km ({params['eps_km']*1000:.0f} meter) — radius cluster<br>
                    • min_samples = {params['min_samples']} — minimum titik untuk membentuk cluster<br>
                    • metric = haversine — jarak di permukaan bumi<br><br>
                    <b>Output:</b><br>
                    • cluster_masuk / cluster_pulang: label cluster (0,1,2,...)<br>
                    • -1 = noise (tidak masuk cluster manapun) → sinyal anomali
                </div>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cluster MASUK",  log.get('step7_clusters_masuk',0))
            with col2:
                st.metric("Noise MASUK",    log.get('step7_noise_masuk',0))
            with col3:
                st.metric("Cluster PULANG", log.get('step7_clusters_pulang',0))
            with col4:
                st.metric("Noise PULANG",   log.get('step7_noise_pulang',0))

        # Step 8
        with st.expander("✅ Step 8: Estimasi Lokasi Kantor per SKPD", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    Mengestimasi koordinat kantor setiap SKPD secara otomatis:<br>
                    1. Ambil data MASUK yang sudah ter-cluster (bukan noise)<br>
                    2. Cari cluster dengan anggota terbanyak per SKPD (= hotspot utama)<br>
                    3. Hitung centroid (rata-rata lat/long) dari cluster terbesar<br>
                    4. Centroid ini digunakan sebagai titik referensi kantor
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Kantor SKPD Terdeteksi", log.get('step8_offices', 0))
            if 'office_centroid' in log:
                st.dataframe(log['office_centroid'].head(10), use_container_width=True)

        # Step 9
        with st.expander("✅ Step 9: Haversine Distance", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    Menghitung jarak setiap titik absensi ke kantor SKPD-nya menggunakan formula Haversine.<br>
                    <b>Mengapa Haversine?</b> Bumi berbentuk bulat, sehingga jarak Euclidean biasa tidak akurat
                    untuk koordinat geografis. Haversine menghitung jarak di permukaan bola.<br><br>
                    <b>Flag yang dibuat:</b><br>
                    • outside_300m: jarak > 300m<br>
                    • very_far: jarak > 5km<br>
                    • extreme_far: jarak > 50km (indikasi fake GPS)
                </div>
            </div>
            """, unsafe_allow_html=True)
            if 'dist_km' in df_result.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rata-rata Jarak", f"{df_result['dist_km'].mean():.3f} km")
                with col2:
                    st.metric("Median Jarak", f"{df_result['dist_km'].median():.3f} km")
                with col3:
                    st.metric("Di luar 300m", f"{df_result['outside_300m'].sum():,}")
                with col4:
                    st.metric("Extreme >50km", f"{df_result['extreme_far'].sum():,}")

        # Step 10
        with st.expander("✅ Step 10: Validation Logic", expanded=False):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    Membuat flag validasi berdasarkan kombinasi jarak + catatan + status_lokasi:<br>
                    • <b>no_note</b>: tidak ada catatan/alasan<br>
                    • <b>far_no_note</b>: jauh + tidak ada alasan → indikasi fraud kuat<br>
                    • <b>far_with_note</b>: jauh tapi ada alasan → kemungkinan tugas luar valid<br>
                    • <b>near_but_status0</b>: dekat tapi sistem lama bilang di luar → mismatch kecil
                </div>
            </div>
            """, unsafe_allow_html=True)
            if 'far_no_note' in df_result.columns:
                val_data = {
                    'Flag': ['far_no_note','far_with_note','near_but_status0'],
                    'Jumlah': [
                        int(df_result['far_no_note'].sum()),
                        int(df_result['far_with_note'].sum()),
                        int(df_result['near_but_status0'].sum())
                    ]
                }
                st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)

        # Step 11
        with st.expander("✅ Step 11: ST-DBSCAN (Spatio-Temporal)", expanded=False):
            st.markdown(f"""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    ST-DBSCAN menggabungkan dimensi spasial (lat/long) dan temporal (jam_desimal)
                    dalam satu clustering.<br><br>
                    <b>Mengapa?</b> Seseorang bisa absen di lokasi yang sama tapi di waktu yang sangat berbeda
                    (misal: absen masuk jam 7 pagi dan jam 11 malam). ST-DBSCAN mendeteksi inkonsistensi ini.<br><br>
                    <b>Parameter:</b><br>
                    • eps_space = {params['st_eps_km']} km<br>
                    • eps_time = {params['st_eps_hours']} jam<br>
                    • min_samples = {params['st_min_samples']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ST-Noise MASUK",  log.get('step12_st_noise_masuk',0))
            with col2:
                st.metric("ST-Noise PULANG", log.get('step12_st_noise_pulang',0))

        # Step 12-13
        with st.expander("✅ Step 12-13: Anomaly Scoring & Risk Level", expanded=True):
            st.markdown("""
            <div class="step-box">
                <div class="step-title">Apa yang dilakukan?</div>
                <div class="step-desc">
                    Menggabungkan semua sinyal anomali menjadi satu skor risiko,
                    lalu mengklasifikasikan ke LOW / MED / HIGH.
                </div>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Data", f"{log.get('step13_total',0):,}")
            with col2:
                st.metric("🔴 HIGH", f"{log.get('step13_high',0):,}")
            with col3:
                st.metric("🟡 MED",  f"{log.get('step13_med',0):,}")
            with col4:
                st.metric("🟢 LOW",  f"{log.get('step13_low',0):,}")

            if 'risk_level' in df_result.columns:
                fig = px.pie(
                    df_result['risk_level'].value_counts().reset_index(),
                    values='count', names='risk_level',
                    title='Distribusi Risk Level',
                    color='risk_level',
                    color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'},
                    hole=0.4
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Download hasil
        st.markdown("---")
        st.markdown("### 💾 Download Hasil Preprocessing")
        csv_buf = io.StringIO()
        df_result.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Download absensi_processed.csv",
            data=csv_buf.getvalue().encode('utf-8'),
            file_name="absensi_processed.csv",
            mime="text/csv",
            use_container_width=True
        )

        excel_buf = io.BytesIO()
        df_result.to_excel(excel_buf, index=False)
        st.download_button(
            label="⬇️ Download absensi_processed.xlsx",
            data=excel_buf.getvalue(),
            file_name="absensi_processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


# ============================================================
# PAGE: VISUALISASI
# ============================================================

def page_visualisasi(filters):
    st.markdown("## 📊 Visualisasi Anomali Absensi")

    if 'df_result' not in st.session_state or st.session_state.df_result is None:
        st.warning("⚠️ Belum ada data. Silakan upload dan jalankan preprocessing terlebih dahulu di menu **📥 Upload & Preprocessing**.")
        return

    df_full = st.session_state.df_result
    log     = st.session_state.get('log', {})

    # Apply filters
    df = apply_filters(
        df_full,
        filters.get('skpd', 'Semua'),
        filters.get('risk', ['HIGH','MED','LOW']),
        filters.get('jenis', ['M','P']),
        filters.get('date', None),
        filters.get('dist', (0.0, 100.0))
    )

    st.caption(f"📊 Menampilkan **{len(df):,}** dari **{len(df_full):,}** total absensi")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "🗺️ Peta", "⏰ Temporal",
        "📏 Jarak", "👤 Karyawan", "🔵 Cluster", "📋 Data"
    ])

    with tab1:
        _tab_overview(df)
    with tab2:
        _tab_map(df, filters, log)
    with tab3:
        _tab_temporal(df)
    with tab4:
        _tab_distance(df)
    with tab5:
        _tab_employee(df)
    with tab6:
        _tab_cluster(df)
    with tab7:
        _tab_data(df)


def _tab_overview(df):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Absensi",  f"{len(df):,}")
    with col2: st.metric("Total Karyawan", f"{df['karyawan_id'].nunique():,}")
    with col3:
        h = (df['risk_level']=='HIGH').sum()
        st.metric("🔴 HIGH", f"{h:,}", delta=f"{h/len(df)*100:.1f}%" if len(df)>0 else "0%")
    with col4: st.metric("🟡 MED",  f"{(df['risk_level']=='MED').sum():,}")
    with col5: st.metric("🟢 LOW",  f"{(df['risk_level']=='LOW').sum():,}")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.pie(df['risk_level'].value_counts().reset_index(),
                     values='count', names='risk_level', title='Distribusi Risk Level',
                     color='risk_level',
                     color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'}, hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        skpd_risk = df.groupby(['id_skpd','risk_level']).size().reset_index(name='count')
        fig = px.bar(skpd_risk, x='id_skpd', y='count', color='risk_level',
                     title='Distribusi Risk per SKPD', barmode='stack',
                     color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'},
                     labels={'id_skpd':'SKPD','count':'Jumlah','risk_level':'Risk'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x='anomaly_score', color='risk_level',
                       title='Distribusi Anomaly Score', nbins=30,
                       color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def _tab_map(df, filters, log):
    if len(df) == 0:
        st.warning("Tidak ada data untuk ditampilkan.")
        return

    MAX_POINTS = 2000
    if len(df) > MAX_POINTS:
        st.info(f"⚠️ Menampilkan {MAX_POINTS} dari {len(df):,} titik untuk performa optimal.")
        df_disp = df.sample(MAX_POINTS, random_state=42)
    else:
        df_disp = df

    office_centroid = log.get('office_centroid', None)
    map_type  = filters.get('map_type', 'marker')
    use_pydeck = filters.get('use_pydeck', False)

    if use_pydeck:
        st.markdown("### 🌐 Peta 3D (PyDeck)")
        deck = create_pydeck_map(df_disp)
        st.pydeck_chart(deck)
    else:
        st.markdown(f"### 🗺️ Peta Interaktif")
        m = create_folium_map(df_disp, map_type, office_centroid)
        st_folium(m, width=None, height=560, returned_objects=[])

    st.markdown("**Legenda:** 🔴 HIGH Risk &nbsp; 🟡 MED Risk &nbsp; 🟢 LOW Risk &nbsp; 🔵 Kantor SKPD &nbsp; ⭕ Radius 300m")


def _tab_temporal(df):
    if 'tanggal_kirim' not in df.columns:
        st.warning("Kolom tanggal_kirim tidak ditemukan.")
        return

    col1, col2 = st.columns(2)
    with col1:
        if 'jam' in df.columns:
            fig = px.bar(df.groupby(['jam','risk_level']).size().reset_index(name='count'),
                         x='jam', y='count', color='risk_level', title='Distribusi per Jam',
                         color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'weekday' in df.columns:
            day_names = {0:'Senin',1:'Selasa',2:'Rabu',3:'Kamis',4:'Jumat',5:'Sabtu',6:'Minggu'}
            df2 = df.copy()
            df2['hari'] = df2['weekday'].map(day_names)
            fig = px.bar(df2.groupby(['hari','risk_level']).size().reset_index(name='count'),
                         x='hari', y='count', color='risk_level', title='Distribusi per Hari',
                         color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'},
                         category_orders={'hari':['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']})
            st.plotly_chart(fig, use_container_width=True)

    if 'tanggal' in df.columns:
        daily = df.groupby(['tanggal','risk_level']).size().reset_index(name='count')
        fig = px.line(daily, x='tanggal', y='count', color='risk_level',
                      title='Trend Anomali Harian', markers=True,
                      color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    if 'jam_desimal' in df.columns and 'dist_km' in df.columns:
        fig = px.scatter(df.sample(min(1000,len(df))), x='jam_desimal', y='dist_km',
                         color='risk_level', title='Jam Absensi vs Jarak ke Kantor',
                         color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'},
                         opacity=0.6, hover_data=['karyawan_id','id_skpd'])
        fig.add_hline(y=0.3, line_dash='dash', line_color='gray', annotation_text='300m')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def _tab_distance(df):
    if 'dist_km' not in df.columns:
        st.warning("Kolom dist_km tidak ditemukan.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Rata-rata Jarak", f"{df['dist_km'].mean():.3f} km")
    with col2: st.metric("Median Jarak",    f"{df['dist_km'].median():.3f} km")
    with col3: st.metric("Jarak Maks",      f"{df['dist_km'].max():.3f} km")
    with col4:
        out = (df['dist_km']>0.3).sum()
        st.metric("Di luar 300m", f"{out:,} ({out/len(df)*100:.1f}%)")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(df[df['dist_km']<=10], x='dist_km', color='risk_level',
                           title='Distribusi Jarak (≤10km)', nbins=50,
                           color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
        fig.add_vline(x=0.3, line_dash='dash', line_color='red', annotation_text='300m')
        fig.add_vline(x=5.0, line_dash='dash', line_color='orange', annotation_text='5km')
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.box(df[df['dist_km']<=10], x='id_skpd', y='dist_km', color='risk_level',
                     title='Distribusi Jarak per SKPD',
                     color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚨 Kasus Extreme (>5km)")
    extreme = df[df['dist_km']>5].sort_values('dist_km', ascending=False)
    if len(extreme) > 0:
        cols_show = ['karyawan_id','id_skpd','jenis','tanggal_kirim','lat','long',
                     'dist_km','anomaly_score','risk_level','system_action','catatan']
        cols_show = [c for c in cols_show if c in extreme.columns]
        st.dataframe(extreme[cols_show].head(50), use_container_width=True)
    else:
        st.success("✅ Tidak ada kasus extreme (>5km) dalam filter ini.")


def _tab_employee(df):
    emp_agg = df.groupby('karyawan_id').agg(
        total_absensi=('karyawan_id','count'),
        avg_dist_km=('dist_km','mean'),
        max_dist_km=('dist_km','max'),
        avg_score=('anomaly_score','mean'),
        max_score=('anomaly_score','max'),
        high_count=('risk_level', lambda x: (x=='HIGH').sum()),
        skpd=('id_skpd','first')
    ).reset_index()
    emp_agg['high_pct'] = (emp_agg['high_count'] / emp_agg['total_absensi'] * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        top = emp_agg.nlargest(10,'high_count')
        fig = px.bar(top, x='karyawan_id', y='high_count', title='Top 10 Karyawan HIGH Risk',
                     color='high_pct', color_continuous_scale='Reds')
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(emp_agg, x='total_absensi', y='avg_score', size='max_dist_km',
                         color='high_pct', hover_data=['karyawan_id','skpd'],
                         title='Total Absensi vs Rata-rata Score',
                         color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚨 Karyawan Berisiko Tinggi")
    risky = emp_agg[emp_agg['high_count']>0].sort_values('high_count', ascending=False)
    if len(risky) > 0:
        st.dataframe(risky.head(30), use_container_width=True)
    else:
        st.success("✅ Tidak ada karyawan dengan HIGH risk dalam filter ini.")


def _tab_cluster(df):
    col1, col2 = st.columns(2)
    with col1:
        if 'cluster_masuk' in df.columns:
            masuk = df[df['jenis']=='M']
            noise_pct = (masuk['cluster_masuk']==-1).mean()*100
            st.metric("Noise MASUK", f"{noise_pct:.1f}%")
            if 'cluster_size_masuk' in df.columns:
                fig = px.histogram(masuk[masuk['cluster_masuk']!=-1], x='cluster_size_masuk',
                                   title='Ukuran Cluster MASUK')
                st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'cluster_pulang' in df.columns:
            pulang = df[df['jenis']=='P']
            noise_pct_p = (pulang['cluster_pulang']==-1).mean()*100
            st.metric("Noise PULANG", f"{noise_pct_p:.1f}%")
            if 'cluster_size_pulang' in df.columns:
                fig = px.histogram(pulang[pulang['cluster_pulang']!=-1], x='cluster_size_pulang',
                                   title='Ukuran Cluster PULANG')
                st.plotly_chart(fig, use_container_width=True)

    if 'is_st_noise_masuk' in df.columns:
        st.markdown("### 🟣 ST-DBSCAN Noise")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("ST-Noise MASUK",  f"{df[df['jenis']=='M']['is_st_noise_masuk'].mean()*100:.1f}%")
        with col4:
            if 'is_st_noise_pulang' in df.columns:
                st.metric("ST-Noise PULANG", f"{df[df['jenis']=='P']['is_st_noise_pulang'].mean()*100:.1f}%")


def _tab_data(df):
    col1, col2 = st.columns(2)
    with col1:
        search = st.text_input("🔍 Cari Karyawan ID", "")
    with col2:
        sort_col = st.selectbox("Urutkan", ['anomaly_score','dist_km','tanggal_kirim'])

    df_t = df.copy()
    if search:
        df_t = df_t[df_t['karyawan_id'].astype(str).str.contains(search)]
    if sort_col in df_t.columns:
        df_t = df_t.sort_values(sort_col, ascending=False)

    cols_show = ['karyawan_id','id_skpd','jenis','tanggal_kirim','lat','long',
                 'dist_km','anomaly_score','risk_level','system_action',
                 'outside_300m','very_far','extreme_far','far_no_note','catatan']
    cols_show = [c for c in cols_show if c in df_t.columns]

    st.dataframe(df_t[cols_show].head(500), use_container_width=True, height=500)
    st.caption(f"Menampilkan {min(500,len(df_t))} dari {len(df_t)} baris")

    csv = df_t[cols_show].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "absensi_filtered.csv", "text/csv")


# ============================================================
# PAGE: PREDIKSI
# ============================================================

def page_prediksi(params):
    st.markdown("## 🔮 Prediksi Anomali Absensi")
    st.markdown("Upload data absensi baru untuk diprediksi apakah termasuk anomali atau tidak.")

    if 'df_result' not in st.session_state or st.session_state.df_result is None:
        st.warning("⚠️ Anda perlu menjalankan preprocessing terlebih dahulu di menu **📥 Upload & Preprocessing** agar sistem memiliki referensi kantor SKPD.")
        return

    log = st.session_state.get('log', {})
    office_centroid = log.get('office_centroid', None)

    if office_centroid is None or len(office_centroid) == 0:
        st.error("❌ Data kantor SKPD tidak tersedia. Jalankan preprocessing terlebih dahulu.")
        return

    st.info(f"✅ Referensi kantor tersedia untuk **{len(office_centroid)} SKPD**")

    # Upload file prediksi
    pred_file = st.file_uploader(
        "Upload file absensi baru untuk diprediksi (CSV/Excel)",
        type=['csv','xlsx'],
        key='pred_upload'
    )

    # Input manual
    st.markdown("### ✏️ Atau Input Manual")
    with st.form("manual_input"):
        col1, col2, col3 = st.columns(3)
        with col1:
            m_karyawan = st.number_input("Karyawan ID", min_value=1, value=1001)
            m_skpd     = st.number_input("ID SKPD", min_value=1, value=int(office_centroid['id_skpd'].iloc[0]))
        with col2:
            m_lat  = st.number_input("Latitude",  value=3.7, format="%.6f")
            m_long = st.number_input("Longitude", value=98.6, format="%.6f")
        with col3:
            m_jenis   = st.selectbox("Jenis", ['M','P'], format_func=lambda x: 'Masuk' if x=='M' else 'Pulang')
            m_waktu   = st.time_input("Jam Absensi")
            m_catatan = st.text_input("Catatan (kosongkan jika tidak ada)", "")

        submitted = st.form_submit_button("🔮 Prediksi", type="primary", use_container_width=True)

    if submitted:
        # Buat dataframe dari input manual
        df_pred = pd.DataFrame([{
            'karyawan_id':   m_karyawan,
            'id_skpd':       m_skpd,
            'lat':           m_lat,
            'long':          m_long,
            'jenis':         m_jenis,
            'tanggal_kirim': pd.Timestamp.now().replace(hour=m_waktu.hour, minute=m_waktu.minute, second=0),
            'catatan':       m_catatan if m_catatan else np.nan,
            'status_lokasi': 1
        }])
        _predict_and_show(df_pred, office_centroid)

    if pred_file is not None:
        if pred_file.name.endswith('.csv'):
            df_pred_file = pd.read_csv(pred_file)
        else:
            df_pred_file = pd.read_excel(pred_file)

        st.markdown(f"### 📋 Prediksi Batch — {len(df_pred_file):,} baris")
        if st.button("🚀 Jalankan Prediksi Batch", type="primary"):
            with st.spinner("⏳ Memproses prediksi..."):
                _predict_and_show(df_pred_file, office_centroid, batch=True)


def _predict_and_show(df_input: pd.DataFrame, office_centroid: pd.DataFrame, batch=False):
    """Prediksi anomali untuk data input baru."""
    df = df_input.copy()

    # Pastikan kolom dasar ada
    if 'tanggal_kirim' in df.columns:
        df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'], errors='coerce')
        df['jam']         = df['tanggal_kirim'].dt.hour
        df['menit']       = df['tanggal_kirim'].dt.minute
        df['jam_desimal'] = df['jam'] + df['menit'] / 60.0
    else:
        df['jam_desimal'] = 8.0

    if 'catatan' not in df.columns:
        df['catatan'] = np.nan
    if 'status_lokasi' not in df.columns:
        df['status_lokasi'] = 1

    # Merge kantor
    df = df.merge(office_centroid, on='id_skpd', how='left')

    # Hitung jarak
    df['dist_km'] = df.apply(
        lambda r: haversine_scalar(r['lat'], r['long'],
                                   r.get('office_lat', r['lat']),
                                   r.get('office_long', r['long']))
        if pd.notna(r.get('office_lat')) else 0.0,
        axis=1
    )

    # Flags
    df['outside_300m']     = (df['dist_km'] > 0.3).astype(int)
    df['very_far']         = (df['dist_km'] > 5.0).astype(int)
    df['extreme_far']      = (df['dist_km'] > 50.0).astype(int)
    df['no_note']          = df['catatan'].isna().astype(int)
    df['far_no_note']      = ((df['outside_300m']==1) & (df['no_note']==1)).astype(int)
    df['far_with_note']    = ((df['outside_300m']==1) & (df['no_note']==0)).astype(int)
    df['near_but_status0'] = ((df['outside_300m']==0) & (df['status_lokasi']==0)).astype(int)

    # Scoring (tanpa cluster karena data baru)
    df['anomaly_score'] = (
        df['outside_300m']   * 25 +
        df['far_no_note']    * 35 +
        df['very_far']       * 30 +
        df['extreme_far']    * 50 +
        df['near_but_status0'] * 5
    )

    def risk_level(s):
        if s >= 70:   return 'HIGH'
        elif s >= 30: return 'MED'
        else:         return 'LOW'

    def system_action(risk):
        if risk == 'LOW':   return '✅ AUTO APPROVE'
        elif risk == 'MED': return '⏸️ HOLD (Perlu Review)'
        else:               return '❌ TEMP REJECT + NOTIF APPROVER'

    df['risk_level']    = df['anomaly_score'].apply(risk_level)
    df['system_action'] = df['risk_level'].apply(system_action)

    if not batch:
        # Tampilan single prediction
        row = df.iloc[0]
        risk  = row['risk_level']
        color = get_risk_color_hex(risk)

        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jarak ke Kantor", f"{row['dist_km']:.3f} km")
        with col2:
            st.metric("Anomaly Score", int(row['anomaly_score']))
        with col3:
            st.metric("Risk Level", risk)

        action = row['system_action']
        if risk == 'LOW':
            st.success(f"**Aksi Sistem:** {action}")
        elif risk == 'MED':
            st.warning(f"**Aksi Sistem:** {action}")
        else:
            st.error(f"**Aksi Sistem:** {action}")

        # Detail sinyal
        st.markdown("**Detail Sinyal Anomali:**")
        signals = {
            'Di luar 300m':        int(row['outside_300m']),
            'Jauh tanpa catatan':  int(row['far_no_note']),
            'Sangat jauh (>5km)':  int(row['very_far']),
            'Extreme (>50km)':     int(row['extreme_far']),
            'Mismatch status':     int(row['near_but_status0']),
        }
        sig_df = pd.DataFrame(list(signals.items()), columns=['Sinyal','Aktif'])
        sig_df['Status'] = sig_df['Aktif'].map({1:'⚠️ YA', 0:'✅ TIDAK'})
        st.dataframe(sig_df[['Sinyal','Status']], use_container_width=True, hide_index=True)

        # Mini peta
        if pd.notna(row.get('office_lat')):
            st.markdown("**Lokasi Absensi vs Kantor:**")
            m = folium.Map(location=[row['lat'], row['long']], zoom_start=15)
            folium.CircleMarker(
                location=[row['lat'], row['long']], radius=10,
                color=color, fill=True, fill_color=color, fill_opacity=0.8,
                popup=f"Absensi — {risk}"
            ).add_to(m)
            folium.Marker(
                location=[row['office_lat'], row['office_long']],
                popup="Kantor SKPD",
                icon=folium.Icon(color='blue', icon='home', prefix='fa')
            ).add_to(m)
            folium.Circle(
                location=[row['office_lat'], row['office_long']],
                radius=300, color='blue', fill=False, weight=2, dash_array='5'
            ).add_to(m)
            folium.PolyLine(
                locations=[[row['lat'], row['long']], [row['office_lat'], row['office_long']]],
                color='gray', weight=2, dash_array='5'
            ).add_to(m)
            st_folium(m, width=None, height=400, returned_objects=[])

    else:
        # Batch prediction
        st.success(f"✅ Prediksi selesai untuk {len(df):,} baris")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("🔴 HIGH", f"{(df['risk_level']=='HIGH').sum():,}")
        with col2: st.metric("🟡 MED",  f"{(df['risk_level']=='MED').sum():,}")
        with col3: st.metric("🟢 LOW",  f"{(df['risk_level']=='LOW').sum():,}")

        fig = px.pie(df['risk_level'].value_counts().reset_index(),
                     values='count', names='risk_level', title='Distribusi Risk Level Prediksi',
                     color='risk_level',
                     color_discrete_map={'HIGH':'#e74c3c','MED':'#f39c12','LOW':'#27ae60'}, hole=0.4)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        cols_show = ['karyawan_id','id_skpd','jenis','tanggal_kirim','lat','long',
                     'dist_km','anomaly_score','risk_level','system_action','catatan']
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(df[cols_show].sort_values('anomaly_score', ascending=False),
                     use_container_width=True)

        csv = df[cols_show].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil Prediksi CSV", csv,
                           "prediksi_anomali.csv", "text/csv", use_container_width=True)


# ============================================================
# MAIN
# ============================================================

def main():
    # Init session state
    if 'df_result' not in st.session_state:
        st.session_state['df_result'] = None
    if 'log' not in st.session_state:
        st.session_state['log'] = {}

    # Sidebar
    page, uploaded, params, filters = render_sidebar()

    # Jika ada file upload baru, simpan ke session state
    if uploaded is not None:
        st.session_state['uploaded_file'] = uploaded

    # Routing halaman
    if page == "🏠 Beranda":
        page_beranda()
    elif page == "📥 Upload & Preprocessing":
        page_preprocessing(uploaded, params)
    elif page == "📊 Visualisasi":
        page_visualisasi(filters)
    elif page == "🔮 Prediksi":
        page_prediksi(params)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:gray;font-size:12px">'
        'Deteksi Anomali Absensi | DBSCAN + ST-DBSCAN + Haversine | Streamlit + Folium + Plotly'
        '</p>', unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
