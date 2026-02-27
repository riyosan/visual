"""
🗺️ Visualisasi Anomali Absensi - Streamlit App
Teknologi: Streamlit + Folium + Plotly + Pydeck
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Visualisasi Anomali Absensi",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high  { color: #e74c3c; font-weight: bold; }
    .risk-med   { color: #f39c12; font-weight: bold; }
    .risk-low   { color: #27ae60; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data(uploaded_file=None):
    """Load data dari file upload atau file lokal."""
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    elif os.path.exists('absensi_processed.csv'):
        df = pd.read_csv('absensi_processed.csv')
    elif os.path.exists('dataset_absensi_final2.xlsx'):
        df = pd.read_excel('dataset_absensi_final2.xlsx')
    else:
        return None

    # Pastikan kolom datetime
    if 'tanggal_kirim' in df.columns:
        df['tanggal_kirim'] = pd.to_datetime(df['tanggal_kirim'])

    # Pastikan kolom risk_level ada
    if 'risk_level' not in df.columns:
        df['risk_level'] = 'LOW'
    if 'anomaly_score' not in df.columns:
        df['anomaly_score'] = 0
    if 'dist_km' not in df.columns:
        df['dist_km'] = 0.0

    return df


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_risk_color(risk: str) -> str:
    return {'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'}.get(risk, '#3498db')


def get_risk_color_folium(risk: str) -> str:
    return {'HIGH': 'red', 'MED': 'orange', 'LOW': 'green'}.get(risk, 'blue')


def create_folium_map(df: pd.DataFrame, map_type: str = 'marker') -> folium.Map:
    """Buat peta Folium dengan berbagai mode tampilan."""
    center_lat  = df['lat'].median()
    center_long = df['long'].median()

    m = folium.Map(
        location=[center_lat, center_long],
        zoom_start=13,
        tiles='OpenStreetMap'
    )

    # Tambahkan layer tile alternatif
    folium.TileLayer('CartoDB positron', name='CartoDB Light').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)

    if map_type == 'heatmap':
        # Heatmap berdasarkan anomaly score
        heat_data = []
        for _, row in df.iterrows():
            weight = row.get('anomaly_score', 1) + 1
            heat_data.append([row['lat'], row['long'], weight])
        HeatMap(heat_data, radius=15, blur=10, max_zoom=14).add_to(m)

    elif map_type == 'cluster':
        # Marker cluster
        mc = MarkerCluster(name='Absensi').add_to(m)
        for _, row in df.iterrows():
            color = get_risk_color_folium(row.get('risk_level', 'LOW'))
            popup_html = _build_popup(row)
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=280)
            ).add_to(mc)

    else:  # marker default
        # Layer per risk level
        for risk in ['HIGH', 'MED', 'LOW']:
            fg = folium.FeatureGroup(name=f'Risk {risk}')
            subset = df[df['risk_level'] == risk]
            color = get_risk_color_folium(risk)
            for _, row in subset.iterrows():
                popup_html = _build_popup(row)
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=6 if risk == 'LOW' else 8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=280)
                ).add_to(fg)
            fg.add_to(m)

    # Marker kantor
    if 'office_lat' in df.columns and 'office_long' in df.columns:
        offices = df[['id_skpd', 'office_lat', 'office_long']].drop_duplicates()
        office_fg = folium.FeatureGroup(name='📍 Kantor SKPD')
        for _, row in offices.iterrows():
            if pd.notna(row['office_lat']):
                folium.Marker(
                    location=[row['office_lat'], row['office_long']],
                    popup=f"<b>Kantor SKPD {row['id_skpd']}</b>",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa')
                ).add_to(office_fg)
                # Lingkaran 300m
                folium.Circle(
                    location=[row['office_lat'], row['office_long']],
                    radius=300,
                    color='blue',
                    fill=False,
                    weight=2,
                    dash_array='5',
                    popup='Radius 300m'
                ).add_to(office_fg)
        office_fg.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _build_popup(row: pd.Series) -> str:
    risk = row.get('risk_level', 'N/A')
    color = get_risk_color(risk)
    dist  = row.get('dist_km', 0)
    score = row.get('anomaly_score', 0)
    catatan = row.get('catatan', '-')
    if pd.isna(catatan):
        catatan = '-'

    return f"""
    <div style='font-family:Arial; font-size:12px; min-width:220px'>
        <h4 style='margin:0 0 8px 0; color:#2c3e50'>📋 Detail Absensi</h4>
        <table style='width:100%; border-collapse:collapse'>
            <tr><td><b>Karyawan</b></td><td>{row.get('karyawan_id','N/A')}</td></tr>
            <tr><td><b>SKPD</b></td><td>{row.get('id_skpd','N/A')}</td></tr>
            <tr><td><b>Jenis</b></td><td>{'🟢 Masuk' if row.get('jenis')=='M' else '🔴 Pulang'}</td></tr>
            <tr><td><b>Waktu</b></td><td>{str(row.get('tanggal_kirim','N/A'))[:16]}</td></tr>
            <tr><td><b>Jarak ke kantor</b></td><td>{dist:.3f} km</td></tr>
            <tr><td><b>Anomaly Score</b></td><td>{score}</td></tr>
            <tr><td><b>Risk Level</b></td>
                <td><span style='color:{color}; font-weight:bold'>{risk}</span></td></tr>
            <tr><td><b>Catatan</b></td><td>{catatan}</td></tr>
        </table>
    </div>
    """


def create_pydeck_map(df: pd.DataFrame) -> pdk.Deck:
    """Buat peta 3D dengan PyDeck (ScatterplotLayer + HexagonLayer)."""
    color_map = {
        'HIGH': [231, 76, 60, 200],
        'MED':  [243, 156, 18, 200],
        'LOW':  [39, 174, 96, 200]
    }
    df_map = df.copy()
    df_map['color'] = df_map['risk_level'].map(color_map).fillna([52, 152, 219, 200])
    df_map['radius'] = df_map['anomaly_score'].fillna(0) * 10 + 20

    scatter_layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_map,
        get_position='[long, lat]',
        get_color='color',
        get_radius='radius',
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=6,
        radius_min_pixels=4,
        radius_max_pixels=30,
    )

    hex_layer = pdk.Layer(
        'HexagonLayer',
        data=df_map,
        get_position='[long, lat]',
        radius=100,
        elevation_scale=4,
        elevation_range=[0, 300],
        pickable=True,
        extruded=True,
        coverage=1,
    )

    view_state = pdk.ViewState(
        latitude=df['lat'].median(),
        longitude=df['long'].median(),
        zoom=12,
        pitch=40,
        bearing=0
    )

    return pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        tooltip={
            'html': '<b>Karyawan:</b> {karyawan_id}<br>'
                    '<b>Risk:</b> {risk_level}<br>'
                    '<b>Score:</b> {anomaly_score}<br>'
                    '<b>Jarak:</b> {dist_km} km',
            'style': {'backgroundColor': 'steelblue', 'color': 'white'}
        }
    )


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("⚙️ Filter & Pengaturan")

    st.sidebar.markdown("### 📂 Upload Data")
    uploaded = st.sidebar.file_uploader(
        "Upload file CSV/Excel hasil preprocessing",
        type=['csv', 'xlsx'],
        help="Upload file absensi_processed.csv dari Google Colab"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filter Data")

    # Filter SKPD
    skpd_list = ['Semua'] + sorted(df['id_skpd'].unique().tolist())
    selected_skpd = st.sidebar.selectbox("SKPD", skpd_list)

    # Filter Risk Level
    risk_options = st.sidebar.multiselect(
        "Risk Level",
        options=['HIGH', 'MED', 'LOW'],
        default=['HIGH', 'MED', 'LOW']
    )

    # Filter Jenis Absensi
    jenis_options = st.sidebar.multiselect(
        "Jenis Absensi",
        options=['M', 'P'],
        default=['M', 'P'],
        format_func=lambda x: 'Masuk' if x == 'M' else 'Pulang'
    )

    # Filter Tanggal
    if 'tanggal_kirim' in df.columns:
        min_date = df['tanggal_kirim'].min().date()
        max_date = df['tanggal_kirim'].max().date()
        date_range = st.sidebar.date_input(
            "Rentang Tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = None

    # Filter Jarak
    max_dist = float(df['dist_km'].max()) if 'dist_km' in df.columns else 100.0
    dist_range = st.sidebar.slider(
        "Jarak ke Kantor (km)",
        min_value=0.0,
        max_value=min(max_dist, 100.0),
        value=(0.0, min(max_dist, 100.0)),
        step=0.1
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗺️ Pengaturan Peta")
    map_type = st.sidebar.radio(
        "Tipe Peta",
        options=['marker', 'cluster', 'heatmap'],
        format_func=lambda x: {
            'marker': '📍 Marker per Risk',
            'cluster': '🔵 Cluster Marker',
            'heatmap': '🔥 Heatmap'
        }[x]
    )

    use_pydeck = st.sidebar.checkbox("Gunakan Peta 3D (PyDeck)", value=False)

    return uploaded, selected_skpd, risk_options, jenis_options, date_range, dist_range, map_type, use_pydeck


def apply_filters(df, selected_skpd, risk_options, jenis_options, date_range, dist_range):
    """Terapkan semua filter ke dataframe."""
    filtered = df.copy()

    if selected_skpd != 'Semua':
        filtered = filtered[filtered['id_skpd'] == selected_skpd]

    if risk_options:
        filtered = filtered[filtered['risk_level'].isin(risk_options)]

    if jenis_options:
        filtered = filtered[filtered['jenis'].isin(jenis_options)]

    if date_range and len(date_range) == 2 and 'tanggal_kirim' in filtered.columns:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['tanggal_kirim'].dt.date >= start_date) &
            (filtered['tanggal_kirim'].dt.date <= end_date)
        ]

    if 'dist_km' in filtered.columns:
        filtered = filtered[
            (filtered['dist_km'] >= dist_range[0]) &
            (filtered['dist_km'] <= dist_range[1])
        ]

    return filtered


# ============================================================
# TABS
# ============================================================
def tab_overview(df_filtered: pd.DataFrame, df_full: pd.DataFrame):
    """Tab ringkasan statistik."""
    st.markdown("## 📊 Ringkasan Data")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Absensi", f"{len(df_filtered):,}")
    with col2:
        st.metric("Total Karyawan", f"{df_filtered['karyawan_id'].nunique():,}")
    with col3:
        high_count = (df_filtered['risk_level'] == 'HIGH').sum()
        st.metric("🔴 HIGH Risk", f"{high_count:,}",
                  delta=f"{high_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%")
    with col4:
        med_count = (df_filtered['risk_level'] == 'MED').sum()
        st.metric("🟡 MED Risk", f"{med_count:,}")
    with col5:
        low_count = (df_filtered['risk_level'] == 'LOW').sum()
        st.metric("🟢 LOW Risk", f"{low_count:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Pie chart risk level
        risk_counts = df_filtered['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Jumlah']
        fig_pie = px.pie(
            risk_counts,
            values='Jumlah',
            names='Risk Level',
            title='Distribusi Risk Level',
            color='Risk Level',
            color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
            hole=0.4
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # Bar chart per SKPD
        skpd_risk = df_filtered.groupby(['id_skpd', 'risk_level']).size().reset_index(name='count')
        fig_bar = px.bar(
            skpd_risk,
            x='id_skpd',
            y='count',
            color='risk_level',
            title='Distribusi Risk per SKPD',
            color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
            barmode='stack',
            labels={'id_skpd': 'SKPD', 'count': 'Jumlah', 'risk_level': 'Risk'}
        )
        fig_bar.update_layout(height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Anomaly score distribution
    fig_hist = px.histogram(
        df_filtered,
        x='anomaly_score',
        color='risk_level',
        title='Distribusi Anomaly Score',
        color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
        nbins=20,
        labels={'anomaly_score': 'Anomaly Score', 'count': 'Jumlah'}
    )
    fig_hist.update_layout(height=300)
    st.plotly_chart(fig_hist, use_container_width=True)


def tab_map(df_filtered: pd.DataFrame, map_type: str, use_pydeck: bool):
    """Tab visualisasi peta."""
    st.markdown("## 🗺️ Peta Visualisasi Absensi")

    if len(df_filtered) == 0:
        st.warning("Tidak ada data untuk ditampilkan. Sesuaikan filter.")
        return

    # Batasi jumlah titik untuk performa
    MAX_POINTS = 2000
    if len(df_filtered) > MAX_POINTS:
        st.info(f"⚠️ Menampilkan {MAX_POINTS} dari {len(df_filtered)} titik untuk performa optimal.")
        df_display = df_filtered.sample(MAX_POINTS, random_state=42)
    else:
        df_display = df_filtered

    if use_pydeck:
        st.markdown("### 🌐 Peta 3D (PyDeck)")
        deck = create_pydeck_map(df_display)
        st.pydeck_chart(deck)
    else:
        st.markdown(f"### 🗺️ Peta Interaktif ({map_type.title()})")
        m = create_folium_map(df_display, map_type)
        map_data = st_folium(m, width=None, height=550, returned_objects=["last_object_clicked"])

        # Info klik
        if map_data and map_data.get("last_object_clicked"):
            clicked = map_data["last_object_clicked"]
            st.info(f"📍 Koordinat diklik: Lat {clicked.get('lat', 'N/A'):.6f}, Long {clicked.get('lng', 'N/A'):.6f}")

    # Legenda
    st.markdown("""
    **Legenda:**
    🔴 HIGH Risk &nbsp;&nbsp; 🟡 MED Risk &nbsp;&nbsp; 🟢 LOW Risk &nbsp;&nbsp; 🔵 Kantor SKPD &nbsp;&nbsp; ⭕ Radius 300m
    """)


def tab_time_analysis(df_filtered: pd.DataFrame):
    """Tab analisis temporal."""
    st.markdown("## ⏰ Analisis Temporal")

    if 'tanggal_kirim' not in df_filtered.columns:
        st.warning("Kolom tanggal_kirim tidak ditemukan.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Absensi per jam
        if 'jam' in df_filtered.columns:
            jam_risk = df_filtered.groupby(['jam', 'risk_level']).size().reset_index(name='count')
            fig_jam = px.bar(
                jam_risk,
                x='jam',
                y='count',
                color='risk_level',
                title='Distribusi Absensi per Jam',
                color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
                labels={'jam': 'Jam', 'count': 'Jumlah'}
            )
            st.plotly_chart(fig_jam, use_container_width=True)

    with col2:
        # Absensi per hari
        if 'weekday' in df_filtered.columns:
            day_names = {0:'Senin', 1:'Selasa', 2:'Rabu', 3:'Kamis', 4:'Jumat', 5:'Sabtu', 6:'Minggu'}
            df_day = df_filtered.copy()
            df_day['hari'] = df_day['weekday'].map(day_names)
            day_risk = df_day.groupby(['hari', 'risk_level']).size().reset_index(name='count')
            day_order = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
            fig_day = px.bar(
                day_risk,
                x='hari',
                y='count',
                color='risk_level',
                title='Distribusi Absensi per Hari',
                color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
                category_orders={'hari': day_order},
                labels={'hari': 'Hari', 'count': 'Jumlah'}
            )
            st.plotly_chart(fig_day, use_container_width=True)

    # Trend harian
    if 'tanggal' in df_filtered.columns:
        daily = df_filtered.groupby(['tanggal', 'risk_level']).size().reset_index(name='count')
        fig_trend = px.line(
            daily,
            x='tanggal',
            y='count',
            color='risk_level',
            title='Trend Anomali Harian',
            color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
            markers=True,
            labels={'tanggal': 'Tanggal', 'count': 'Jumlah'}
        )
        fig_trend.update_layout(height=350)
        st.plotly_chart(fig_trend, use_container_width=True)

    # Scatter jam vs jarak
    if 'jam_desimal' in df_filtered.columns and 'dist_km' in df_filtered.columns:
        fig_scatter = px.scatter(
            df_filtered.sample(min(1000, len(df_filtered))),
            x='jam_desimal',
            y='dist_km',
            color='risk_level',
            title='Jam Absensi vs Jarak ke Kantor',
            color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
            opacity=0.6,
            labels={'jam_desimal': 'Jam (desimal)', 'dist_km': 'Jarak (km)'},
            hover_data=['karyawan_id', 'id_skpd']
        )
        fig_scatter.add_hline(y=0.3, line_dash='dash', line_color='gray',
                               annotation_text='Batas 300m')
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)


def tab_distance_analysis(df_filtered: pd.DataFrame):
    """Tab analisis jarak."""
    st.markdown("## 📏 Analisis Jarak ke Kantor")

    if 'dist_km' not in df_filtered.columns:
        st.warning("Kolom dist_km tidak ditemukan.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rata-rata Jarak", f"{df_filtered['dist_km'].mean():.3f} km")
    with col2:
        st.metric("Median Jarak", f"{df_filtered['dist_km'].median():.3f} km")
    with col3:
        st.metric("Jarak Maks", f"{df_filtered['dist_km'].max():.3f} km")
    with col4:
        outside = (df_filtered['dist_km'] > 0.3).sum()
        st.metric("Di luar 300m", f"{outside:,} ({outside/len(df_filtered)*100:.1f}%)")

    col_left, col_right = st.columns(2)

    with col_left:
        # Histogram jarak
        fig_dist = px.histogram(
            df_filtered[df_filtered['dist_km'] <= 10],  # Batasi 10km untuk keterbacaan
            x='dist_km',
            color='risk_level',
            title='Distribusi Jarak ke Kantor (≤10km)',
            color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
            nbins=50,
            labels={'dist_km': 'Jarak (km)', 'count': 'Jumlah'}
        )
        fig_dist.add_vline(x=0.3, line_dash='dash', line_color='red',
                            annotation_text='300m')
        fig_dist.add_vline(x=5.0, line_dash='dash', line_color='orange',
                            annotation_text='5km')
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        # Box plot jarak per SKPD
        fig_box = px.box(
            df_filtered[df_filtered['dist_km'] <= 10],
            x='id_skpd',
            y='dist_km',
            color='risk_level',
            title='Distribusi Jarak per SKPD',
            color_discrete_map={'HIGH': '#e74c3c', 'MED': '#f39c12', 'LOW': '#27ae60'},
            labels={'id_skpd': 'SKPD', 'dist_km': 'Jarak (km)'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Tabel extreme cases
    st.markdown("### 🚨 Kasus Extreme (>5km)")
    extreme = df_filtered[df_filtered['dist_km'] > 5].sort_values('dist_km', ascending=False)
    if len(extreme) > 0:
        cols_show = ['karyawan_id', 'id_skpd', 'jenis', 'tanggal_kirim',
                     'lat', 'long', 'dist_km', 'anomaly_score', 'risk_level', 'catatan']
        cols_show = [c for c in cols_show if c in extreme.columns]
        st.dataframe(
            extreme[cols_show].head(50).style.background_gradient(
                subset=['dist_km', 'anomaly_score'], cmap='Reds'
            ),
            use_container_width=True
        )
    else:
        st.success("✅ Tidak ada kasus extreme (>5km) dalam filter ini.")


def tab_employee_analysis(df_filtered: pd.DataFrame):
    """Tab analisis per karyawan."""
    st.markdown("## 👤 Analisis per Karyawan")

    # Agregasi per karyawan
    emp_agg = df_filtered.groupby('karyawan_id').agg(
        total_absensi=('karyawan_id', 'count'),
        avg_dist_km=('dist_km', 'mean'),
        max_dist_km=('dist_km', 'max'),
        avg_score=('anomaly_score', 'mean'),
        max_score=('anomaly_score', 'max'),
        high_count=('risk_level', lambda x: (x == 'HIGH').sum()),
        med_count=('risk_level', lambda x: (x == 'MED').sum()),
        skpd=('id_skpd', 'first')
    ).reset_index()

    emp_agg['high_pct'] = (emp_agg['high_count'] / emp_agg['total_absensi'] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        # Top 10 karyawan dengan HIGH risk terbanyak
        top_high = emp_agg.nlargest(10, 'high_count')
        fig_top = px.bar(
            top_high,
            x='karyawan_id',
            y='high_count',
            title='Top 10 Karyawan - HIGH Risk Terbanyak',
            color='high_pct',
            color_continuous_scale='Reds',
            labels={'karyawan_id': 'Karyawan ID', 'high_count': 'Jumlah HIGH Risk'}
        )
        fig_top.update_xaxes(type='category')
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        # Scatter: total absensi vs avg score
        fig_emp = px.scatter(
            emp_agg,
            x='total_absensi',
            y='avg_score',
            size='max_dist_km',
            color='high_pct',
            hover_data=['karyawan_id', 'skpd', 'max_dist_km'],
            title='Total Absensi vs Rata-rata Anomaly Score',
            color_continuous_scale='RdYlGn_r',
            labels={
                'total_absensi': 'Total Absensi',
                'avg_score': 'Rata-rata Anomaly Score',
                'high_pct': '% HIGH Risk'
            }
        )
        st.plotly_chart(fig_emp, use_container_width=True)

    # Tabel karyawan berisiko tinggi
    st.markdown("### 🚨 Karyawan Berisiko Tinggi")
    risky = emp_agg[emp_agg['high_count'] > 0].sort_values('high_count', ascending=False)
    if len(risky) > 0:
        st.dataframe(
            risky.head(30).style.background_gradient(
                subset=['high_count', 'avg_score', 'max_dist_km'], cmap='Reds'
            ),
            use_container_width=True
        )
    else:
        st.success("✅ Tidak ada karyawan dengan HIGH risk dalam filter ini.")


def tab_cluster_analysis(df_filtered: pd.DataFrame):
    """Tab analisis cluster DBSCAN."""
    st.markdown("## 🔵 Analisis Cluster DBSCAN")

    col1, col2 = st.columns(2)

    with col1:
        if 'cluster_masuk' in df_filtered.columns:
            masuk_df = df_filtered[df_filtered['jenis'] == 'M']
            noise_pct = (masuk_df['cluster_masuk'] == -1).mean() * 100
            st.metric("Noise MASUK", f"{noise_pct:.1f}%",
                      help="Persentase absensi masuk yang tidak masuk cluster manapun")

            # Distribusi cluster size
            if 'cluster_size_masuk' in df_filtered.columns:
                fig_cs = px.histogram(
                    masuk_df[masuk_df['cluster_masuk'] != -1],
                    x='cluster_size_masuk',
                    title='Distribusi Ukuran Cluster MASUK',
                    labels={'cluster_size_masuk': 'Ukuran Cluster', 'count': 'Jumlah Cluster'}
                )
                st.plotly_chart(fig_cs, use_container_width=True)

    with col2:
        if 'cluster_pulang' in df_filtered.columns:
            pulang_df = df_filtered[df_filtered['jenis'] == 'P']
            noise_pct_p = (pulang_df['cluster_pulang'] == -1).mean() * 100
            st.metric("Noise PULANG", f"{noise_pct_p:.1f}%",
                      help="Persentase absensi pulang yang tidak masuk cluster manapun")

            if 'cluster_size_pulang' in df_filtered.columns:
                fig_cp = px.histogram(
                    pulang_df[pulang_df['cluster_pulang'] != -1],
                    x='cluster_size_pulang',
                    title='Distribusi Ukuran Cluster PULANG',
                    labels={'cluster_size_pulang': 'Ukuran Cluster', 'count': 'Jumlah Cluster'}
                )
                st.plotly_chart(fig_cp, use_container_width=True)

    # ST-DBSCAN noise
    if 'is_st_noise_masuk' in df_filtered.columns:
        st.markdown("### 🟣 ST-DBSCAN Noise Analysis")
        col3, col4 = st.columns(2)
        with col3:
            st_noise_m = df_filtered[df_filtered['jenis']=='M']['is_st_noise_masuk'].mean() * 100
            st.metric("ST-Noise MASUK", f"{st_noise_m:.1f}%",
                      help="Tidak konsisten secara ruang-waktu")
        with col4:
            if 'is_st_noise_pulang' in df_filtered.columns:
                st_noise_p = df_filtered[df_filtered['jenis']=='P']['is_st_noise_pulang'].mean() * 100
                st.metric("ST-Noise PULANG", f"{st_noise_p:.1f}%")


def tab_data_table(df_filtered: pd.DataFrame):
    """Tab tabel data lengkap."""
    st.markdown("## 📋 Tabel Data")

    # Filter tambahan
    col1, col2 = st.columns(2)
    with col1:
        search_id = st.text_input("🔍 Cari Karyawan ID", "")
    with col2:
        sort_col = st.selectbox(
            "Urutkan berdasarkan",
            options=['anomaly_score', 'dist_km', 'tanggal_kirim'],
            index=0
        )

    df_table = df_filtered.copy()
    if search_id:
        df_table = df_table[df_table['karyawan_id'].astype(str).str.contains(search_id)]

    if sort_col in df_table.columns:
        df_table = df_table.sort_values(sort_col, ascending=False)

    # Kolom yang ditampilkan
    display_cols = [
        'karyawan_id', 'id_skpd', 'jenis', 'tanggal_kirim',
        'lat', 'long', 'dist_km', 'anomaly_score', 'risk_level',
        'outside_300m', 'very_far', 'extreme_far',
        'far_no_note', 'far_with_note', 'catatan'
    ]
    display_cols = [c for c in display_cols if c in df_table.columns]

    st.dataframe(
        df_table[display_cols].head(500),
        use_container_width=True,
        height=500
    )

    st.caption(f"Menampilkan {min(500, len(df_table))} dari {len(df_table)} baris")

    # Download
    csv = df_table[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="absensi_filtered.csv",
        mime="text/csv"
    )


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<div class="main-header">🗺️ Visualisasi Anomali Absensi</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:gray">Deteksi Anomali Lokasi Absensi Pegawai menggunakan DBSCAN & ST-DBSCAN</p>', unsafe_allow_html=True)

    # Load data awal
    df_raw = load_data()

    # Sidebar
    (uploaded, selected_skpd, risk_options, jenis_options,
     date_range, dist_range, map_type, use_pydeck) = render_sidebar(
        df_raw if df_raw is not None else pd.DataFrame()
    )

    # Reload jika ada upload
    if uploaded is not None:
        df_raw = load_data(uploaded)
        st.cache_data.clear()

    if df_raw is None or len(df_raw) == 0:
        st.warning("""
        ⚠️ **Data belum tersedia!**

        Silakan:
        1. Jalankan notebook `preprocessing_absensi.ipynb` di Google Colab
        2. Export hasilnya sebagai `absensi_processed.csv`
        3. Upload file tersebut melalui sidebar kiri

        Atau letakkan file `absensi_processed.csv` / `dataset_absensi_final2.xlsx`
        di folder yang sama dengan `app.py`
        """)
        return

    # Apply filters
    df_filtered = apply_filters(df_raw, selected_skpd, risk_options,
                                 jenis_options, date_range, dist_range)

    st.caption(f"📊 Menampilkan **{len(df_filtered):,}** dari **{len(df_raw):,}** total absensi")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "🗺️ Peta",
        "⏰ Temporal",
        "📏 Jarak",
        "👤 Karyawan",
        "🔵 Cluster",
        "📋 Data"
    ])

    with tab1:
        tab_overview(df_filtered, df_raw)
    with tab2:
        tab_map(df_filtered, map_type, use_pydeck)
    with tab3:
        tab_time_analysis(df_filtered)
    with tab4:
        tab_distance_analysis(df_filtered)
    with tab5:
        tab_employee_analysis(df_filtered)
    with tab6:
        tab_cluster_analysis(df_filtered)
    with tab7:
        tab_data_table(df_filtered)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:gray; font-size:12px">'
        '🗺️ Visualisasi Anomali Absensi | DBSCAN + ST-DBSCAN + Haversine | Streamlit + Folium + Plotly'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
