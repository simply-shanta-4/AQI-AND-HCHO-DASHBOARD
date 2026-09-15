# ==============================================================================
# UNIFIED ATMOSPHERIC & PM2.5 SPATIAL INTELLIGENCE PLATFORM (HIGH-PERFORMANCE)
# Ultra-Fast In-Memory Caching & GPU/WebGL Optimized Rendering
# ==============================================================================

import os
import sys
import json
from pathlib import Path
from datetime import datetime, date, timedelta

import requests as _requests_lib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------------
# PATH CONFIGURATION & IMPORTS
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
HCHO_DIR = ROOT_DIR / "hcho database" / "HCHO DETECTION"
SIH_DIR = ROOT_DIR / "SIH_Final" / "SIH_Final"
API_BASE_URL = "https://2232-2409-40d0-3024-c2b6-f523-f9ce-a66b-dc38.ngrok-free.app"

if str(HCHO_DIR) not in sys.path:
    sys.path.insert(0, str(HCHO_DIR))
if str(SIH_DIR) not in sys.path:
    sys.path.insert(0, str(SIH_DIR))

# Location metadata
try:
    from location_metadata import CLUSTER_LOCATIONS, get_location_name
except ImportError:
    CLUSTER_LOCATIONS = {}
    def get_location_name(cid):
        return f"Cluster #{cid}"

# SIH PM2.5 model loader
try:
    from inference.model_loader import ProductionArtifacts, load_production_artifacts
except Exception:
    load_production_artifacts = None

# ------------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AirTrace AI • Unified Atmospheric & PM2.5 Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# MODERN BENTO SAAS CSS STYLING
# ------------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Page Styling */
    html, body, [class*="css"], .stApp, section.main, .block-container {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F8F9FA !important;
        color: #111827 !important;
    }

    /* Dark Typography */
    p, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #111827 !important;
    }
    .stCaption, caption, small {
        color: #6B7280 !important;
    }
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1440px;
    }

    /* Sidebar Background and Typography */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #ECEEF1 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #1F2937 !important;
    }

    /* Radio navigation buttons in sidebar */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 5px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: #F9FAFB !important;
        border: 1px solid #F3F4F6 !important;
        padding: 8px 12px !important;
        border-radius: 12px !important;
        transition: all 0.1s ease !important;
        cursor: pointer !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: #F3F4F6 !important;
        border-color: #E5E7EB !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label span {
        color: #1F2937 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: #E6F4EA !important;
        border: 1px solid #A7F3D0 !important;
        border-left: 4px solid #115E59 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] span {
        color: #0F5132 !important;
        font-weight: 700 !important;
    }

    /* Inputs, Selectboxes, Multiselects */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] * {
        color: #111827 !important;
        background-color: transparent !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
    }
    ul[role="listbox"] li, div[data-baseweb="menu"] * {
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }
    ul[role="listbox"] li:hover, div[data-baseweb="menu"] *:hover {
        background-color: #F3F4F6 !important;
        color: #115E59 !important;
    }
    div[data-baseweb="tag"] {
        background-color: #E5E7EB !important;
    }
    div[data-baseweb="tag"] * {
        color: #111827 !important;
    }
    input[type="text"], input[type="number"], input[type="search"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 10px !important;
    }

    /* Top Navigation Header */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
        background: #FFFFFF;
        padding: 10px 22px;
        border-radius: 16px;
        border: 1px solid #ECEEF1;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .search-pill {
        display: flex;
        align-items: center;
        background: #F3F4F6;
        padding: 7px 14px;
        border-radius: 10px;
        color: #4B5563 !important;
        font-size: 0.86rem;
        width: 340px;
        gap: 8px;
    }
    .user-profile-badge {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .user-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #E0E7FF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.05rem;
    }
    
    /* Page Title & Actions */
    .page-title-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 1.2rem;
    }
    .page-main-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #111827 !important;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.90rem;
        color: #6B7280 !important;
        margin-top: 3px;
        margin-bottom: 0;
    }
    .btn-solid-green {
        background: #115E59;
        color: #FFFFFF !important;
        padding: 7px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.82rem;
        border: none;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        text-decoration: none;
    }
    .btn-outline {
        background: #FFFFFF;
        color: #1F2937 !important;
        padding: 7px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #D1D5DB;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        text-decoration: none;
    }

    /* Bento Cards */
    .bento-card {
        background: #FFFFFF;
        border: 1px solid #ECEEF1;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.02);
        height: 100%;
        position: relative;
    }
    .bento-card-green {
        background: linear-gradient(135deg, #134E3F 0%, #0D382D 100%);
        border: 1px solid #134E3F;
        border-radius: 18px;
        padding: 20px;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(19, 78, 63, 0.16);
        height: 100%;
        position: relative;
    }
    .bento-card-blue {
        background: linear-gradient(135deg, #1E3A8A 0%, #172554 100%);
        border: 1px solid #1E3A8A;
        border-radius: 18px;
        padding: 20px;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.16);
        height: 100%;
        position: relative;
    }
    .bento-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .bento-title {
        font-size: 0.90rem;
        font-weight: 600;
        color: #4B5563 !important;
    }
    .bento-title-light {
        font-size: 0.90rem;
        font-weight: 600;
        color: #E6F4EA !important;
    }
    .arrow-icon-btn {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.80rem;
        color: #374151 !important;
    }
    .arrow-icon-btn-light {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.80rem;
        color: #FFFFFF !important;
    }
    .bento-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #111827 !important;
        line-height: 1.1;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .bento-value-light {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.1;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .bento-subtext {
        font-size: 0.80rem;
        color: #16A34A !important;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .bento-subtext-light {
        font-size: 0.80rem;
        color: #86EFAC !important;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* List Item Row */
    .list-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 9px 0;
        border-bottom: 1px solid #F3F4F6;
    }
    .list-row:last-child {
        border-bottom: none;
    }
    .list-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .list-icon {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.90rem;
    }
    .list-title {
        font-size: 0.84rem;
        font-weight: 700;
        color: #1F2937 !important;
        margin-bottom: 2px;
    }
    .list-subtitle {
        font-size: 0.74rem;
        color: #9CA3AF !important;
    }
    .badge-pill-strong {
        background: #DCFCE7 !important;
        color: #15803D !important;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 20px;
    }
    .badge-pill-moderate {
        background: #FEF3C7 !important;
        color: #B45309 !important;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 20px;
    }
    .badge-pill-weak {
        background: #FEE2E2 !important;
        color: #B91C1C !important;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 20px;
    }

    /* Tracker Card */
    .tracker-card {
        background: linear-gradient(145deg, #093327 0%, #031D16 100%);
        border-radius: 18px;
        padding: 20px;
        color: #FFFFFF !important;
        height: 100%;
        position: relative;
    }
    .tracker-digits {
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin: 8px 0;
        color: #FFFFFF !important;
    }
    .tracker-card * {
        color: #FFFFFF !important;
    }

    /* Sidebar Brand & Promo */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.20rem;
        font-weight: 800;
        color: #111827 !important;
        margin-bottom: 18px;
    }
    .sidebar-section-title {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #6B7280 !important;
        text-transform: uppercase;
        margin-top: 16px;
        margin-bottom: 6px;
    }
    .sidebar-promo-box {
        background: linear-gradient(135deg, #0E3E32 0%, #06231B 100%) !important;
        border-radius: 14px;
        padding: 14px;
        color: #FFFFFF !important;
        margin-top: 20px;
    }
    .sidebar-promo-box * {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# DOMESTIC INDIAN HOTSPOT REGIONS (28 CLUSTERS)
# ------------------------------------------------------------------------------
LOCAL_INDIAN_REGIONS = {cid: info for cid, info in CLUSTER_LOCATIONS.items() if info.get('country') == 'India'}

# ------------------------------------------------------------------------------
# GRANULAR CACHED DATA LOADERS (SUB-MILLISECOND ACCESS)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_hcho_report():
    features_dir = HCHO_DIR / "features"
    report_file = features_dir / "FINAL_SOURCE_REGION_REPORT_2025.csv"
    if not report_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(report_file)
    indian_cluster_ids = list(LOCAL_INDIAN_REGIONS.keys())
    df['cluster_id'] = df['cluster_id'].astype(int)
    df = df[df['cluster_id'].isin(indian_cluster_ids)].copy()
    df['location_name'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['location_name'])
    df['state'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['state'])
    df['zone'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['zone'])
    df['district'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['district'])
    df['region_type'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['region_type'])
    df['description'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['description'])
    df['full_location'] = df['location_name'] + ", " + df['state']
    df['map_label'] = "#" + df['validation_rank'].astype(str) + " " + df['location_name']
    return df

@st.cache_data(show_spinner=False)
def get_hcho_monthly():
    features_dir = HCHO_DIR / "features"
    monthly_file = features_dir / "SOURCE_RECEPTOR_TRANSPORT_MONTHLY_2025.csv"
    if not monthly_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(monthly_file)
    indian_cluster_ids = list(LOCAL_INDIAN_REGIONS.keys())
    df['cluster_id'] = df['cluster_id'].astype(int)
    df = df[df['cluster_id'].isin(indian_cluster_ids)].copy()
    df['location_name'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['location_name'])
    df['state'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['state'])
    df['zone'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['zone'])
    df['cluster_label'] = "#" + df['cluster_id'].astype(str) + " " + df['location_name']
    return df

@st.cache_data(show_spinner=False)
def get_hcho_sampled_points(frac=0.5):
    features_dir = HCHO_DIR / "features"
    hdbscan_pts_file = features_dir / "HCHO_SOURCE_CLUSTERS_HDBSCAN_2025.csv"
    if not hdbscan_pts_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(hdbscan_pts_file)
    indian_cluster_ids = list(LOCAL_INDIAN_REGIONS.keys())
    df['cluster_id'] = df['cluster_id'].astype(int)
    df = df[df['cluster_id'].isin(indian_cluster_ids)].copy()
    df['location_name'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['location_name'])
    df['state'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['state'])
    df['zone'] = df['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['zone'])
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=42)
    return df

@st.cache_resource(show_spinner=False)
def get_sih_artifacts():
    if load_production_artifacts is not None:
        try:
            return load_production_artifacts(SIH_DIR)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def get_sih_benchmarks():
    extras_dir = SIH_DIR / "extras"
    config_dir = SIH_DIR / "config"
    blend_file = extras_dir / "validation_blend_search.csv"
    rep_tail_file = extras_dir / "representative_test_tail_metrics.csv"
    meta_file = config_dir / "production_metadata.json"
    
    df_blend = pd.read_csv(blend_file) if blend_file.exists() else pd.DataFrame()
    df_rep_tail = pd.read_csv(rep_tail_file) if rep_tail_file.exists() else pd.DataFrame()
    metadata = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return df_blend, df_rep_tail, metadata

@st.cache_data(show_spinner=False)
def load_tiff_preview():
    """Load one HCHO TIFF as base64 PNG for aesthetic satellite display."""
    tiff_path = HCHO_DIR / "HCHO" / "HCHO_2025_01.tif"
    if not tiff_path.exists():
        return None
    try:
        import rasterio
        import io as _io
        import base64 as _b64
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt

        with rasterio.open(tiff_path) as src:
            data = src.read(1).astype(float)
            nd = src.nodata
            if nd is not None:
                data[data == nd] = np.nan

        fig, ax = _plt.subplots(figsize=(5, 3.2), dpi=130)
        valid = data[np.isfinite(data)]
        if len(valid) > 0:
            vmin, vmax = np.nanpercentile(valid, [2, 98])
            im = ax.imshow(data, cmap='cividis', vmin=vmin, vmax=vmax, aspect='auto', interpolation='bilinear')
            cbar = _plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
            cbar.ax.tick_params(labelsize=6)
        ax.axis('off')
        _plt.tight_layout(pad=0.3)
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#FAFBFC', edgecolor='none')
        _plt.close(fig)
        buf.seek(0)
        return _b64.b64encode(buf.read()).decode()
    except Exception:
        return None

# Instant prediction calculation with caching
@st.cache_data(show_spinner=False)
def compute_pm25_prediction(c_lat, c_lon, target_date_str, model_blend_weight):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    lat_dist_delhi = np.sqrt((c_lat - 28.6)**2 + (c_lon - 77.2)**2)
    is_winter = target_dt.month in [10, 11, 12, 1, 2]
    base_pm = 185.0 if (lat_dist_delhi < 2.5 and is_winter) else (115.0 if is_winter else 60.0)
    
    np.random.seed(int(c_lat * 100 + c_lon * 10 + target_dt.day))
    sim_raw = max(15.0, base_pm + np.random.normal(0, 15.0))
    sim_extreme = max(20.0, sim_raw * 1.15 + (35.0 if is_winter else 0.0))
    
    final_pm = model_blend_weight * sim_raw + (1.0 - model_blend_weight) * sim_extreme
    return float(sim_raw), float(sim_extreme), float(final_pm)

# Pre-load lightweight datasets
df_report = get_hcho_report()
df_monthly = get_hcho_monthly()
df_blend, df_rep_tail, sih_metadata = get_sih_benchmarks()
hcho_features_dir = HCHO_DIR / "features"

# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="background: #115E59; color: white; width: 34px; height: 34px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; font-size: 1.1rem;">🛰️</span>
        <span style="font-weight: 800; font-size: 1.15rem; color: #111827;">AirTrace AI</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">INTELLIGENCE DOMAIN</div>', unsafe_allow_html=True)
    domain_mode = st.radio(
        "Select System Domain",
        options=[
            "🌿 Formaldehyde & VOC Observatory",
            "🧠 AI PM2.5 Spatial Prediction Engine",
            "🌐 Cross-Pollutant Multi-Sensor Hub"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="sidebar-section-title">MODULE NAVIGATION</div>', unsafe_allow_html=True)
    
    if domain_mode == "🌿 Formaldehyde & VOC Observatory":
        app_subpage = st.radio(
            "HCHO Sub-Navigation",
            options=[
                "📊 Dashboard Overview",
                "🗺️ Local Hotspot Map",
                "📈 Transport & Dynamics",
                "🔬 Grid Points & Clusters",
                "📑 Report & Downloads",
                "🖼️ Publication Figures"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="sidebar-section-title">DOMESTIC FILTERS</div>', unsafe_allow_html=True)
        all_states = sorted(df_report['state'].dropna().unique().tolist()) if 'state' in df_report else []
        selected_states = st.multiselect("Select State / Region:", options=all_states, default=all_states)
        all_zones = sorted(df_report['zone'].dropna().unique().tolist()) if 'zone' in df_report else []
        selected_zones = st.multiselect("Geographic Zone:", options=all_zones, default=all_zones)
        val_classes = sorted(df_report['validation_class'].dropna().unique().tolist()) if 'validation_class' in df_report else []
        selected_classes = st.multiselect("Validation Confidence:", options=val_classes, default=val_classes)

    elif domain_mode == "🧠 AI PM2.5 Spatial Prediction Engine":
        app_subpage = st.radio(
            "PM2.5 Sub-Navigation",
            options=[
                "🔮 Interactive PM2.5 Prediction",
                "🛰️ Multi-Sensor Satellite Explorer",
                "📊 Model Benchmarks & Tail Stress",
                "🏗️ Deep Learning Architecture & Pipeline"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="sidebar-section-title">PREDICTION CONTROLS</div>', unsafe_allow_html=True)
        enable_aod = st.toggle("Include MOSDAC AOD Diagnostics", value=False)

    else:
        app_subpage = st.radio(
            "Cross-Domain Sub-Navigation",
            options=[
                "🔗 VOC Precursor & PM2.5 Coupling",
                "🔥 Biomass Fire Radiative Dynamics",
                "📋 Regulatory Policy & NCAP Action Matrix"
            ],
            label_visibility="collapsed"
        )

# Apply global sidebar filters
filtered_report = df_report.copy()
if domain_mode == "🌿 Formaldehyde & VOC Observatory":
    if selected_states:
        filtered_report = filtered_report[filtered_report['state'].isin(selected_states)]
    if selected_zones:
        filtered_report = filtered_report[filtered_report['zone'].isin(selected_zones)]
    if selected_classes:
        filtered_report = filtered_report[filtered_report['validation_class'].isin(selected_classes)]

# ------------------------------------------------------------------------------
# TOP NAVBAR
# ------------------------------------------------------------------------------
active_badge_title = "India VOC Hotspot Observatory" if domain_mode.startswith("🌿") else ("Deep Learning PM2.5 Spatial Engine" if domain_mode.startswith("🧠") else "Atmospheric Cross-Correlation Hub")
active_badge_sub = "28 Domestic Clustered Regions • 2025" if domain_mode.startswith("🌿") else ("Dual CNN-BiLSTM Blend • Sentinel-5P + MOSDAC" if domain_mode.startswith("🧠") else "Multi-Sensor Remote Sensing & Chemical Transport")

st.markdown(f"""
<div class="top-navbar">
    <div class="search-pill">
        <span>🔍</span>
        <span>Search states, industrial zones, Delhi NCR coordinates...</span>
        <span style="margin-left: auto; background: #E5E7EB; padding: 2px 6px; border-radius: 6px; font-size: 0.7rem; font-weight: 700;">⌘ K</span>
    </div>
    <div class="user-profile-badge">
        <span style="font-size: 1.05rem; cursor: pointer; margin-right: 4px;">📡</span>
        <span style="font-size: 1.05rem; cursor: pointer; margin-right: 8px;">📊</span>
        <div class="user-avatar">🇮🇳</div>
        <div>
            <div style="font-weight: 700; font-size: 0.84rem; color: #111827; line-height: 1.1;">{active_badge_title}</div>
            <div style="font-size: 0.72rem; color: #6B7280;">{active_badge_sub}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# DOMAIN 1: FORMALDEHYDE (HCHO) TRACE & VOC OBSERVATORY
# ==============================================================================
if domain_mode == "🌿 Formaldehyde & VOC Observatory":

    # --- VIEW 1.1: DASHBOARD OVERVIEW ---
    if app_subpage == "📊 Dashboard Overview":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">India Formaldehyde Hotspots Dashboard</h1>
                <p class="page-subtitle">Domestic monitoring of reactive NMVOC proxies, seasonal stubble burning, and petrochemical chemical corridors (2025).</p>
            </div>
            <div style="display: flex; gap: 10px;">
                <a class="btn-solid-green" href="javascript:void(0);">🌿 28 Source Regions</a>
                <a class="btn-outline" href="javascript:void(0);">Export Report</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        total_count = len(filtered_report)
        strong_count = len(filtered_report[filtered_report['validation_class'] == 'Strong'])
        mod_count = len(filtered_report[filtered_report['validation_class'] == 'Moderate'])
        weak_count = len(filtered_report[filtered_report['validation_class'] == 'Weak'])
        
        with kpi_col1:
            st.markdown(f"""
            <div class="bento-card-green">
                <div class="bento-header">
                    <span class="bento-title-light">Total Indian Regions</span>
                    <div class="arrow-icon-btn-light">↗</div>
                </div>
                <div class="bento-value-light">{total_count}</div>
                <div class="bento-subtext-light">🌿 Across 14 Indian States</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-header">
                    <span class="bento-title">Strong Downwind Validation</span>
                    <div class="arrow-icon-btn">↗</div>
                </div>
                <div class="bento-value">{strong_count}</div>
                <div class="bento-subtext">🟢 Statistically Proven Plumes</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-header">
                    <span class="bento-title">Moderate Confidence</span>
                    <div class="arrow-icon-btn">↗</div>
                </div>
                <div class="bento-value">{mod_count}</div>
                <div class="bento-subtext" style="color:#D97706;">🟡 Consistent Transport</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-header">
                    <span class="bento-title">Emerging / Diffuse</span>
                    <div class="arrow-icon-btn">↗</div>
                </div>
                <div class="bento-value">{weak_count}</div>
                <div class="bento-subtext" style="color:#6B7280;">🟠 Localized / Seasonal Burning</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        mid_col1, mid_col2, mid_col3 = st.columns([1.4, 1.2, 1.4])
        
        with mid_col1:
            st.markdown("""
            <div class="bento-card">
                <div style="font-weight: 800; font-size: 1.02rem; color: #111827; margin-bottom: 10px;">Domestic Emission Dynamics (Monthly Fire FRP)</div>
            """, unsafe_allow_html=True)
            
            if not df_monthly.empty:
                monthly_agg = df_monthly.groupby('month').agg({'source_FRP': 'sum', 'mean_downwind_HCHO': 'mean'}).reset_index()
                bar_colors = ['#C8E6C9', '#81C784', '#4CAF50', '#2E7D32', '#1B5E20', '#388E3C', '#66BB6A', '#A5D6A7', '#C8E6C9', '#E8F5E9', '#81C784', '#4CAF50']
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[f"M{m}" for m in monthly_agg['month']],
                    y=monthly_agg['source_FRP'] / 1000.0,
                    marker=dict(color=bar_colors[:len(monthly_agg)], line=dict(width=0), opacity=0.9),
                    hovertext=[f"Month {m}: {frp:,.0f} kMW" for m, frp in zip(monthly_agg['month'], monthly_agg['source_FRP'])],
                    hoverinfo="text"
                ))
                fig_bar.update_layout(
                    height=205,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color='#9CA3AF')),
                    yaxis=dict(showgrid=True, gridcolor='#F3F4F6', zeroline=False, tickfont=dict(size=10, color='#9CA3AF')),
                    bargap=0.35
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

        with mid_col2:
            top_hotspot = filtered_report.sort_values(by='validation_rank').iloc[0] if not filtered_report.empty else {}
            st.markdown(f"""
            <div class="bento-card" style="display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="font-weight: 800; font-size: 1.02rem; color: #111827; margin-bottom: 8px;">Top Local Hotspot</div>
                    <div style="font-size: 1.02rem; font-weight: 800; color: #115E59; margin-bottom: 4px;">#{top_hotspot.get('validation_rank', 1)} {top_hotspot.get('location_name', 'Sangli-Satara')}</div>
                    <div style="font-size: 0.80rem; color: #6B7280; margin-bottom: 8px;">📍 {top_hotspot.get('state', 'Maharashtra')} ({top_hotspot.get('zone', 'West Zone')})</div>
                    <div style="font-size: 0.76rem; color: #4B5563; background:#F0FDF4; padding:8px 12px; border-radius:10px; border:1px solid #DCFCE7;">
                        ⚡ <b>Downwind Ratio:</b> {top_hotspot.get('observed_downwind_upwind_ratio', 1.17):.3f}x<br>
                        🔥 <b>Total Fire FRP:</b> {top_hotspot.get('total_FRP', 0):,.0f} MW<br>
                        🏭 <b>Type:</b> {top_hotspot.get('region_type', 'Agro-Industrial')}
                    </div>
                </div>
                <div style="margin-top: 12px;">
                    <div style="font-size:0.75rem; color:#115E59; font-weight:700; background:#E6F4EA; padding:6px 10px; border-radius:8px; text-align:center;">
                        High-Confidence True Emission Plume
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with mid_col3:
            st.markdown("""
            <div class="bento-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: 800; font-size: 1.02rem; color: #111827;">Top Ranked Indian Hotspots</span>
                    <span style="background: #F3F4F6; color: #374151; padding: 3px 8px; border-radius: 8px; font-size: 0.70rem; font-weight: 700;">Ranked</span>
                </div>
            """, unsafe_allow_html=True)
            
            for _, r in filtered_report.sort_values(by='validation_rank').head(4).iterrows():
                vc = r['validation_class']
                badge_class = "badge-pill-strong" if vc == "Strong" else ("badge-pill-moderate" if vc == "Moderate" else "badge-pill-weak")
                st.markdown(f"""
                <div class="list-row">
                    <div class="list-left">
                        <div class="list-icon" style="background: #ECFDF5; color: #047857;">📍</div>
                        <div>
                            <div class="list-title">#{r['validation_rank']} {r['location_name']}</div>
                            <div class="list-subtitle">{r['state']} • {r['zone']}</div>
                        </div>
                    </div>
                    <div>
                        <span class="{badge_class}">{vc}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        bot_col1, bot_col2, bot_col3 = st.columns([1.4, 1.2, 1.4])
        
        with bot_col1:
            st.markdown("""
            <div class="bento-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: 800; font-size: 1.02rem; color: #111827;">Hotspot Count by State</span>
                    <span style="background: #F3F4F6; color: #374151; padding: 3px 8px; border-radius: 8px; font-size: 0.70rem; font-weight: 700;">States</span>
                </div>
            """, unsafe_allow_html=True)
            
            state_counts = filtered_report['state'].value_counts()
            for sname, count in state_counts.head(5).items():
                st.markdown(f"""
                <div class="list-row">
                    <div class="list-left">
                        <span style="font-size: 1.05rem;">🏛️</span>
                        <div>
                            <div class="list-title">{sname}</div>
                            <div class="list-subtitle">{count} Clustered Source Regions</div>
                        </div>
                    </div>
                    <div>
                        <span class="badge-pill-strong" style="background:#F3F4F6; color:#374151;">{count} Hotspots</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with bot_col2:
            st.markdown("""
            <div class="bento-card">
                <div style="font-weight: 800; font-size: 1.02rem; color: #111827; margin-bottom: 6px;">Domestic Validation Progress</div>
            """, unsafe_allow_html=True)
            
            validated_pct = (strong_count + mod_count) / max(1, total_count) * 100.0
            
            fig_gauge = go.Figure(data=[go.Pie(
                values=[strong_count, mod_count, weak_count],
                labels=['Strong', 'Moderate', 'Weak'],
                hole=.72,
                marker_colors=['#134E3F', '#34D399', '#E5E7EB'],
                textinfo='none',
                sort=False,
                direction='clockwise'
            )])
            fig_gauge.update_layout(
                height=175,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                annotations=[dict(text=f"<b>{validated_pct:.0f}%</b><br><span style='font-size:11px;color:#9CA3AF;'>Validated</span>", x=0.5, y=0.5, font_size=19, showarrow=False)]
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

        with bot_col3:
            total_cells_india = int(filtered_report['cells'].sum()) if 'cells' in filtered_report else 0
            st.markdown(f"""
            <div class="tracker-card">
                <div style="font-weight: 700; font-size: 0.86rem; color: #86EFAC;">Subcontinent Spatial Monitoring Sync</div>
                <div class="tracker-digits">{total_cells_india:,}</div>
                <div style="font-size: 0.78rem; opacity: 0.85;">High-Density Spatial Grid Cells Monitored across India</div>
                <div style="margin-top: 12px; display: flex; gap: 8px;">
                    <span style="background: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight:700;">Sentinel-5P L3</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight:700;">ERA5 Wind</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- VIEW 1.2: INTERACTIVE LOCAL HOTSPOT MAP ---
    elif app_subpage == "🗺️ Local Hotspot Map":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">Interactive Local Hotspots Map</h1>
                <p class="page-subtitle">Geospatial exploration of 28 domestic Indian source regions across agricultural zones, industrial corridors, and coal basins.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_preset, col_view_opt, col_label_opt = st.columns([2, 1, 1])
        preset_options = [
            "🇮🇳 All India Overview",
            "🌾 Northern Region (Punjab, UP, Rajasthan)",
            "🏭 Western Region (Maharashtra & Gujarat)",
            "⚡ Central Energy Hub (MP, Chhattisgarh)",
            "🌴 Southern Peninsula (TN, KA, AP, Kerala)",
            "🌲 Eastern & North-Eastern (Bihar, WB, Jharkhand, Assam)"
        ]
        
        cluster_targets = [f"🎯 #{int(r['validation_rank'])} {r['location_name']} ({r['state']})" for _, r in df_report.sort_values(by='validation_rank').iterrows()]
        
        with col_preset:
            selected_focus = st.selectbox("📍 Quick Zoom / Fly-To Region:", options=preset_options + ["--- Local Hotspots ---"] + cluster_targets, index=0)
        with col_view_opt:
            map_style = st.selectbox("🗺️ Base Map Layer:", options=["open-street-map", "carto-positron", "carto-darkmatter", "white-bg"], index=0)
        with col_label_opt:
            show_labels = st.checkbox("🏷️ Show District Labels", value=True)
            
        col_map_opt1, col_map_opt2 = st.columns(2)
        with col_map_opt1:
            color_metric = st.selectbox("Color Markers By:", options=["validation_class", "zone", "final_source_score", "total_FRP", "total_fire_count", "observed_downwind_upwind_ratio", "mean_wind_speed"], format_func=lambda x: x.replace('_', ' ').title())
        with col_map_opt2:
            size_metric = st.selectbox("Size Markers By:", options=["cells", "total_FRP", "total_fire_count", "final_source_score"], format_func=lambda x: x.replace('_', ' ').title())

        center_lat, center_lon, zoom_level = 21.0, 78.5, 4.4
        selected_cid_detail = None
        
        if selected_focus == "🌾 Northern Region (Punjab, UP, Rajasthan)":
            center_lat, center_lon, zoom_level = 27.5, 78.0, 5.8
        elif selected_focus == "🏭 Western Region (Maharashtra & Gujarat)":
            center_lat, center_lon, zoom_level = 20.5, 74.5, 5.6
        elif selected_focus == "⚡ Central Energy Hub (MP, Chhattisgarh)":
            center_lat, center_lon, zoom_level = 23.0, 80.5, 5.8
        elif selected_focus == "🌴 Southern Peninsula (TN, KA, AP, Kerala)":
            center_lat, center_lon, zoom_level = 13.5, 78.0, 5.5
        elif selected_focus == "🌲 Eastern & North-Eastern (Bihar, WB, Jharkhand, Assam)":
            center_lat, center_lon, zoom_level = 24.5, 88.0, 5.6
        elif "🎯" in selected_focus:
            try:
                rank_str = selected_focus.split(" ")[1].replace("#", "").strip()
                rank_val = int(rank_str)
                target_row = df_report[df_report['validation_rank'] == rank_val]
                if not target_row.empty:
                    center_lat = float(target_row.iloc[0]['mean_latitude'])
                    center_lon = float(target_row.iloc[0]['mean_longitude'])
                    zoom_level = 8.5
                    selected_cid_detail = int(target_row.iloc[0]['cluster_id'])
            except Exception:
                pass

        if not filtered_report.empty:
            hover_data = {
                'location_name': True, 'state': True, 'zone': True, 'district': True,
                'validation_rank': True, 'validation_class': True, 'cells': True,
                'total_fire_count': True, 'total_FRP': ':.1f',
                'observed_downwind_upwind_ratio': ':.3f', 'final_source_score': ':.3f',
                'mean_latitude': ':.3f', 'mean_longitude': ':.3f'
            }
            color_discrete_map = {"Strong": "#10B981", "Moderate": "#F59E0B", "Weak": "#EF4444"}

            fig_map = px.scatter_mapbox(
                filtered_report,
                lat="mean_latitude",
                lon="mean_longitude",
                color=color_metric,
                size=size_metric,
                size_max=30,
                zoom=zoom_level,
                center={"lat": center_lat, "lon": center_lon},
                mapbox_style=map_style,
                color_discrete_map=color_discrete_map if color_metric == "validation_class" else None,
                color_continuous_scale="Plasma" if color_metric not in ["validation_class", "zone", "state"] else None,
                text="map_label" if show_labels else None,
                hover_name="full_location",
                hover_data=hover_data,
                height=600
            )
            fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})
            
            # Selected Location Detail Box
            st.markdown("#### 📍 Selected Hotspot Profile")
            inspect_cid = selected_cid_detail if selected_cid_detail is not None else int(filtered_report.iloc[0]['cluster_id'])
            chosen_info = df_report[df_report['cluster_id'] == inspect_cid].iloc[0]
            val_color = "#10B981" if chosen_info['validation_class'] == "Strong" else ("#F59E0B" if chosen_info['validation_class'] == "Moderate" else "#EF4444")
            
            st.markdown(f"""
            <div class="bento-card" style="background:#F0FDF4; border: 1px solid #DCFCE7;">
                <h3 style="margin-top:0; color:#115E59;">📍 {chosen_info['location_name']}, {chosen_info['state']}</h3>
                <p style="color:#374151;"><b>Key Districts:</b> {chosen_info['district']} • <b>Zone:</b> {chosen_info['zone']}</p>
                <p style="color:#374151;"><b>Landscape & Driver:</b> {chosen_info['description']}</p>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-top: 10px;">
                    <div><b>Validation Rank:</b> #{chosen_info['validation_rank']} in India</div>
                    <div><b>Confidence:</b> <span style="color:{val_color}; font-weight:700;">{chosen_info['validation_class']}</span></div>
                    <div><b>Coordinates:</b> {chosen_info['mean_latitude']:.3f}° N, {chosen_info['mean_longitude']:.3f}° E</div>
                    <div><b>Spatial Cells:</b> {int(chosen_info['cells']):,}</div>
                    <div><b>Total Fire Count:</b> {int(chosen_info['total_fire_count']):,} fires</div>
                    <div><b>Total Fire FRP:</b> {chosen_info['total_FRP']:,.1f} MW</div>
                    <div><b>Downwind Ratio:</b> {chosen_info['observed_downwind_upwind_ratio']:.3f}x</div>
                    <div><b>Wind Speed:</b> {chosen_info['mean_wind_speed']:.2f} m/s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No hotspots match current filters.")

    # --- VIEW 1.3: TRANSPORT & DYNAMICS ---
    elif app_subpage == "📈 Transport & Dynamics":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">Transport & Dynamics</h1>
                <p class="page-subtitle">Monthly wind transport verification and downwind vs upwind plume enhancements across Indian regions.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if not df_monthly.empty:
            col_m1, col_m2 = st.columns([1, 3])
            with col_m1:
                cluster_choice_list = [f"#{int(r['validation_rank'])} {r['location_name']} ({r['state']})" for _, r in df_report.sort_values(by='validation_rank').iterrows()]
                selected_choice = st.selectbox("Select Hotspot Location:", options=cluster_choice_list, index=0)
                selected_rank = int(selected_choice.split(" ")[0].replace("#", ""))
                cluster_info = df_report[df_report['validation_rank'] == selected_rank]
                selected_cluster = int(cluster_info.iloc[0]['cluster_id']) if not cluster_info.empty else int(df_report.iloc[0]['cluster_id'])
                
                if not cluster_info.empty:
                    c_row = cluster_info.iloc[0]
                    st.markdown(f"""
                    <div class="bento-card">
                        <b>📍 Location:</b> {c_row.get('location_name')}<br>
                        <b>🏛️ State:</b> {c_row.get('state')}<br>
                        <b>🌐 Zone:</b> {c_row.get('zone')}<br>
                        <b>🏆 Rank:</b> #{c_row.get('validation_rank', 'N/A')}<br>
                        <b>🛡️ Validation:</b> {c_row.get('validation_class', 'N/A')}<br>
                        <b>🔥 Total FRP:</b> {c_row.get('total_FRP', 0):,.1f} MW<br>
                        <b>⭐ Final Score:</b> {c_row.get('final_source_score', 0):.3f}
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_m2:
                cluster_monthly = df_monthly[df_monthly['cluster_id'] == selected_cluster].sort_values(by='month')
                c_name = cluster_info.iloc[0]['location_name'] if not cluster_info.empty else f"Cluster {selected_cluster}"
                
                fig_monthly = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=(f"{c_name}: Downwind vs Upwind HCHO Concentrations", f"{c_name}: Monthly Fire FRP & Wind Speed"),
                    specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
                )
                fig_monthly.add_trace(go.Scatter(x=cluster_monthly['month'], y=cluster_monthly['mean_downwind_HCHO'], name="Downwind HCHO", line=dict(color="#EF4444", width=3)), row=1, col=1)
                fig_monthly.add_trace(go.Scatter(x=cluster_monthly['month'], y=cluster_monthly['mean_upwind_HCHO'], name="Upwind HCHO", line=dict(color="#10B981", width=2, dash="dash")), row=1, col=1)
                fig_monthly.add_trace(go.Bar(x=cluster_monthly['month'], y=cluster_monthly['source_FRP'], name="Fire FRP (MW)", marker_color="#115E59", opacity=0.7), row=2, col=1, secondary_y=False)
                fig_monthly.add_trace(go.Scatter(x=cluster_monthly['month'], y=cluster_monthly['mean_wind_speed'], name="Wind Speed (m/s)", line=dict(color="#06B6D4", width=2.5)), row=2, col=1, secondary_y=True)
                
                fig_monthly.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
                st.plotly_chart(fig_monthly, use_container_width=True, config={'displayModeBar': False})

    # --- VIEW 1.4: GRID POINTS & CLUSTERS ---
    elif app_subpage == "🔬 Grid Points & Clusters":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">Grid Points & HDBSCAN Clusters</h1>
                <p class="page-subtitle">Candidate grid cells across the Indian subcontinent clustered via HDBSCAN density clustering.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_g1, col_g2 = st.columns([1, 3])
        with col_g1:
            sample_rate = st.slider("Sample fraction (%):", min_value=10, max_value=100, value=30, step=10)
            color_choice = st.selectbox("Color Cells By:", options=["state", "zone", "location_name", "mean_HCHO", "total_FRP", "cluster_probability"])
            
        sampled_df = get_hcho_sampled_points(frac=sample_rate/100.0)
        with col_g2:
            if not sampled_df.empty:
                if color_choice in ["state", "zone", "location_name"]:
                    fig_cells = px.scatter(sampled_df, x="longitude", y="latitude", color=color_choice, title=f"Spatial Point Cloud ({len(sampled_df):,} sampled cells)", height=480, render_mode="webgl")
                else:
                    fig_cells = px.scatter(sampled_df, x="longitude", y="latitude", color=color_choice, color_continuous_scale="Turbo", title=f"Spatial Point Cloud ({len(sampled_df):,} sampled cells)", height=480, render_mode="webgl")
                fig_cells.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cells, use_container_width=True, config={'displayModeBar': False})

    # --- VIEW 1.5: REPORT & DOWNLOADS ---
    elif app_subpage == "📑 Report & Downloads":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">India Domestic Report & Data Export</h1>
                <p class="page-subtitle">Complete catalog of 28 validated Indian formaldehyde source regions with multi-sensor attributes.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        search_query = st.text_input("🔍 Search Table (by state, district, or region name):", "")
        display_df = filtered_report.copy()
        if search_query:
            display_df = display_df[display_df['location_name'].astype(str).str.contains(search_query, case=False) | display_df['state'].astype(str).str.contains(search_query, case=False) | display_df['district'].astype(str).str.contains(search_query, case=False)]
            
        st.dataframe(display_df, use_container_width=True, height=420)
        csv_bytes = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download India Hotspots Master CSV", data=csv_bytes, file_name="India_HCHO_Hotspots_2025.csv", mime="text/csv")

    # --- VIEW 1.6: RESEARCH FIGURES ---
    elif app_subpage == "🖼️ Publication Figures":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">Publication Research Figures</h1>
                <p class="page-subtitle">High-resolution analytical figures, cluster maps, and transport validations.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        figure_catalog = [
            {"title": "1. Final Source Region Map (2025)", "filename": "FINAL_SOURCE_REGION_MAP_2025.png", "desc": "Geographic map showing all validated HCHO source regions color-coded by validation confidence."},
            {"title": "2. Source Region Rankings & Scores", "filename": "FINAL_SOURCE_REGION_RANKING_2025.png", "desc": "Bar chart ranking the top identified hotspot regions based on multi-criteria source scores."},
            {"title": "3. Downwind HCHO Transport Evidence", "filename": "FINAL_TRANSPORT_EVIDENCE_2025.png", "desc": "Empirical evidence demonstrating formaldehyde concentration enhancements downwind."},
            {"title": "4. Downwind Enhancement Ratios", "filename": "FINAL_DOWNWIND_ENHANCEMENT_2025.png", "desc": "Distribution and statistical testing of downwind/upwind HCHO ratios."},
            {"title": "5. Source Region Validation Confidence Map", "filename": "FINAL_SOURCE_REGION_VALIDATION_MAP_2025.png", "desc": "Spatial distribution categorized into Strong, Moderate, and Weak validation evidence."},
            {"title": "6. HDBSCAN Clustered Source Regions", "filename": "FINAL_HDBSCAN_SOURCE_REGIONS_2025.png", "desc": "Density-based spatial cluster boundaries identified across the subcontinent."},
            {"title": "7. Comprehensive HDBSCAN Overview", "filename": "HCHO_SOURCE_REGIONS_HDBSCAN_2025.png", "desc": "High-density multi-panel spatial mapping of all clustered source regions."},
            {"title": "8. HDBSCAN Parameter Sensitivity & Solution Comparison", "filename": "HDBSCAN_SOLUTION_COMPARISON_2025.png", "desc": "Sensitivity testing across hyperparameters."},
            {"title": "9. Spatial Candidate Points Inspection", "filename": "SPATIAL_CANDIDATE_INSPECTION_2025.png", "desc": "Spatial distribution of the raw candidate grid points."}
        ]
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 4, 1])
        if "fig_index" not in st.session_state:
            st.session_state.fig_index = 0
            
        with col_ctrl1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.fig_index = (st.session_state.fig_index - 1) % len(figure_catalog)
        with col_ctrl3:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.fig_index = (st.session_state.fig_index + 1) % len(figure_catalog)
        with col_ctrl2:
            titles = [f["title"] for f in figure_catalog]
            selected_title = st.selectbox("Select Figure:", options=titles, index=st.session_state.fig_index, label_visibility="collapsed")
            st.session_state.fig_index = titles.index(selected_title)
            
        current_fig = figure_catalog[st.session_state.fig_index]
        fig_path = hcho_features_dir / current_fig["filename"]
        
        st.markdown(f"### {current_fig['title']}")
        st.caption(f"{current_fig['desc']} • File: `{current_fig['filename']}`")
        
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
            with open(fig_path, "rb") as file:
                st.download_button(label=f"💾 Download Image ({current_fig['filename']})", data=file, file_name=current_fig["filename"], mime="image/png")


# ==============================================================================
# DOMAIN 2: AI PM2.5 SPATIAL PREDICTION ENGINE (DEEP LEARNING)
# ==============================================================================
elif domain_mode == "🧠 AI PM2.5 Spatial Prediction Engine":

    # --- VIEW 2.1: INTERACTIVE SPATIAL PREDICTION (LIVE API) ---
    if app_subpage == "🔮 Interactive PM2.5 Prediction":
        # Session state
        if 'aqi_pred_result' not in st.session_state:
            st.session_state.aqi_pred_result = None
        if 'aqi_pred_ts' not in st.session_state:
            st.session_state.aqi_pred_ts = None
        if 'aqi_pred_error' not in st.session_state:
            st.session_state.aqi_pred_error = None

        # --- Page Header ---
        _ts_label = ""
        if st.session_state.aqi_pred_ts:
            _delta_s = (datetime.now() - st.session_state.aqi_pred_ts).total_seconds()
            if _delta_s < 60:
                _ts_label = "Updated just now"
            elif _delta_s < 3600:
                _ts_label = f"Updated {int(_delta_s // 60)}m ago"
            else:
                _ts_label = f"Updated {int(_delta_s // 3600)}h ago"

        _status_html = ""
        if st.session_state.aqi_pred_result:
            _status_html = '<span class="badge-pill-strong" style="font-size:0.78rem; padding:5px 12px;">● Prediction Ready</span>'
        elif st.session_state.aqi_pred_error:
            _status_html = '<span class="badge-pill-weak" style="font-size:0.78rem; padding:5px 12px;">● Error</span>'

        _right_side = ""
        if _status_html or _ts_label:
            _ts_div = f'<div style="font-size:0.74rem; color:#9CA3AF; margin-top:4px;">{_ts_label}</div>' if _ts_label else ""
            _right_side = f'<div style="text-align:right;">{_status_html}{_ts_div}</div>'

        st.markdown(f"""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">AQI Prediction</h1>
                <p class="page-subtitle">AI-powered PM2.5 prediction from Sentinel-5P atmospheric observations</p>
            </div>
            {_right_side}
        </div>
        """, unsafe_allow_html=True)

        # --- Input Controls ---
        st.markdown('<style>.stNumberInput button{display:none!important;}</style>', unsafe_allow_html=True)
        _col_sp, _col_cfg = st.columns([1.6, 2.4])

        with _col_sp:
            st.markdown("""
            <div class="bento-card">
                <div style="font-weight:800; font-size:1.0rem; color:#111827; margin-bottom:12px;">📍 Spatial Extent</div>
            """, unsafe_allow_html=True)
            _sc1, _sc2 = st.columns(2)
            with _sc1:
                bbox_west = st.number_input("West (°E)", value=76.83, step=0.01, format="%.2f", key="aqp_w")
                bbox_south = st.number_input("South (°N)", value=28.40, step=0.01, format="%.2f", key="aqp_s")
            with _sc2:
                bbox_east = st.number_input("East (°E)", value=77.34, step=0.01, format="%.2f", key="aqp_e")
                bbox_north = st.number_input("North (°N)", value=28.89, step=0.01, format="%.2f", key="aqp_n")
            st.caption(f"Center: {(bbox_south+bbox_north)/2:.3f}°N, {(bbox_west+bbox_east)/2:.3f}°E")
            st.markdown("</div>", unsafe_allow_html=True)

        with _col_cfg:
            st.markdown("""
            <div class="bento-card">
                <div style="font-weight:800; font-size:1.0rem; color:#111827; margin-bottom:12px;">🗓️ Prediction Configuration</div>
            """, unsafe_allow_html=True)
            _dc1, _dc2 = st.columns(2)
            with _dc1:
                pred_date = st.date_input("Prediction Date", value=date.today(), key="aqp_date")
            with _dc2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                predict_clicked = st.button("🛰️  Run Prediction", type="primary", use_container_width=True, key="aqp_run")

            _valid_input = True
            if bbox_west >= bbox_east:
                st.error("West longitude must be less than East longitude.")
                _valid_input = False
            if bbox_south >= bbox_north:
                st.error("South latitude must be less than North latitude.")
                _valid_input = False
            st.markdown("</div>", unsafe_allow_html=True)

        # --- API Call ---
        if predict_clicked and _valid_input:
            with st.spinner("🛰️ Fetching Sentinel-5P satellite observations & running AI prediction..."):
                try:
                    _payload = {
                        "spatial_extent": {
                            "west": float(bbox_west), "south": float(bbox_south),
                            "east": float(bbox_east), "north": float(bbox_north)
                        },
                        "date": pred_date.strftime("%Y-%m-%d"),
                        "time": "string",
                        "include_aod_diagnostic": enable_aod
                    }
                    _headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
                    _resp = _requests_lib.post(f"{API_BASE_URL}/predict", json=_payload, headers=_headers, timeout=600)

                    if _resp.status_code == 200:
                        st.session_state.aqi_pred_result = _resp.json()
                        st.session_state.aqi_pred_ts = datetime.now()
                        st.session_state.aqi_pred_error = None
                        st.rerun()
                    elif _resp.status_code == 422:
                        st.session_state.aqi_pred_error = "Invalid request parameters. Please check your spatial extent and date."
                        st.session_state.aqi_pred_result = None
                    elif _resp.status_code == 400:
                        st.session_state.aqi_pred_error = "Prediction unavailable for this request."
                        st.session_state.aqi_pred_result = None
                    elif _resp.status_code == 503:
                        st.session_state.aqi_pred_error = "Satellite data service temporarily unavailable. Please try again later."
                        st.session_state.aqi_pred_result = None
                    else:
                        st.session_state.aqi_pred_error = f"Prediction service error (HTTP {_resp.status_code}). Please try again."
                        st.session_state.aqi_pred_result = None
                except _requests_lib.exceptions.Timeout:
                    st.session_state.aqi_pred_error = "Request timed out. Satellite data retrieval may be taking longer than expected."
                    st.session_state.aqi_pred_result = None
                except _requests_lib.exceptions.ConnectionError:
                    st.session_state.aqi_pred_error = "Could not connect to prediction service. Please verify the backend is running."
                    st.session_state.aqi_pred_result = None
                except Exception:
                    st.session_state.aqi_pred_error = "An unexpected error occurred. Please try again."
                    st.session_state.aqi_pred_result = None

        # --- Error Display ---
        if st.session_state.aqi_pred_error:
            st.error(st.session_state.aqi_pred_error)

        # --- Results Dashboard ---
        _res = st.session_state.aqi_pred_result
        if _res:
            _pm25 = _res.get("extreme_model_prediction", 0)
            _raw = _res.get("raw_model_prediction", 0)
            _pm25_ref = _res.get("pm25_ug_m3", 0)
            _date_str = _res.get("date", "")
            _clat = _res.get("center_latitude", 0)
            _clon = _res.get("center_longitude", 0)
            _sp = _res.get("spatial_extent", {})
            _dq = _res.get("data_quality", {})
            _req_d = _dq.get("required_days", 14)
            _use_d = _dq.get("usable_days", 0)
            _miss_d = _dq.get("missing_days", [])
            _miss_c = _dq.get("missing_components", {})
            _seq_s = _res.get("sequence_start", "")
            _seq_e = _res.get("sequence_end", "")
            _seq_len = _res.get("sequence_length", 14)
            _sat_src = _res.get("sentinel_source", "N/A")
            _aod_used = _res.get("aod_used_by_model", False)
            _aod_diag = _res.get("aod_diagnostic")
            _calibrated = _res.get("calibrated", False)
            _notes = _res.get("notes", [])

            # PM2.5 category (NAQI 24hr breakpoints)
            if _pm25 <= 30:
                _cat, _cc = "Good", "#10B981"
            elif _pm25 <= 60:
                _cat, _cc = "Satisfactory", "#34D399"
            elif _pm25 <= 90:
                _cat, _cc = "Moderate", "#F59E0B"
            elif _pm25 <= 120:
                _cat, _cc = "Poor", "#F97316"
            elif _pm25 <= 250:
                _cat, _cc = "Very Poor", "#EF4444"
            else:
                _cat, _cc = "Severe", "#7C3AED"

            # Data quality
            _dq_pct = (_use_d / _req_d * 100) if _req_d > 0 else 0
            if _dq_pct >= 85:
                _dq_lab, _dq_col = "Good", "#10B981"
            elif _dq_pct >= 60:
                _dq_lab, _dq_col = "Partial", "#F59E0B"
            else:
                _dq_lab, _dq_col = "Limited", "#EF4444"

            try:
                _dt_disp = datetime.strptime(_date_str, "%Y-%m-%d").strftime("%d %b %Y")
            except Exception:
                _dt_disp = _date_str

            # ═══════════ HERO PM2.5 CARD ═══════════
            st.markdown(f"""
            <div class="bento-card" style="border-left:4px solid {_cc}; margin:8px 0 16px 0;">
                <div class="bento-header">
                    <div>
                        <span style="font-size:0.76rem; font-weight:700; color:#6B7280; text-transform:uppercase; letter-spacing:0.06em;">AI Prediction · {_dt_disp}</span>
                    </div>
                    <span style="background:{_cc}18; color:{_cc}; font-weight:800; padding:4px 14px; border-radius:20px; font-size:0.82rem;">{_cat}</span>
                </div>
                <div style="display:flex; align-items:baseline; gap:12px; margin:10px 0 6px 0;">
                    <div style="font-size:3.8rem; font-weight:800; color:{_cc}; line-height:1.0; letter-spacing:-0.03em;">{_pm25:.1f}</div>
                    <div style="font-size:1.15rem; color:#6B7280; font-weight:600;">µg/m³</div>
                </div>
                <div style="font-size:0.82rem; color:#6B7280;">
                    Predicted PM2.5 · Delhi ({_clat:.3f}°N, {_clon:.3f}°E)
                </div>
                <div style="margin-top:8px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
                    <span style="font-size:0.72rem; color:#9CA3AF;">NAQI PM2.5 24hr breakpoints</span>
                    <span style="font-size:0.72rem; color:#D1D5DB;">·</span>
                    <span style="font-size:0.72rem; color:#9CA3AF;">Source: extreme_model_prediction</span>
                    <span style="font-size:0.72rem; color:#D1D5DB;">·</span>
                    <span style="font-size:0.72rem; color:#9CA3AF;">● Prediction ready</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ═══════════ ROW 2: Data Quality · Spatial · Model Info ═══════════
            _r2a, _r2b, _r2c = st.columns(3)

            with _r2a:
                _circ = 2 * 3.14159 * 36
                _fill = _circ * (_dq_pct / 100)
                _gap = _circ - _fill
                st.markdown(f"""
                <div class="bento-card">
                    <div class="bento-header">
                        <span class="bento-title">Data Availability</span>
                        <span style="background:{_dq_col}18; color:{_dq_col}; font-weight:700; padding:2px 10px; border-radius:20px; font-size:0.70rem;">{_dq_lab}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:18px; margin:10px 0;">
                        <svg width="90" height="90" viewBox="0 0 90 90">
                            <circle cx="45" cy="45" r="36" fill="none" stroke="#F3F4F6" stroke-width="7"/>
                            <circle cx="45" cy="45" r="36" fill="none" stroke="{_dq_col}" stroke-width="7"
                                stroke-dasharray="{_fill:.1f} {_gap:.1f}" stroke-linecap="round"
                                transform="rotate(-90 45 45)"/>
                            <text x="45" y="42" text-anchor="middle" font-size="15" font-weight="800" fill="#111827">{_use_d}/{_req_d}</text>
                            <text x="45" y="56" text-anchor="middle" font-size="9" fill="#6B7280">days</text>
                        </svg>
                        <div>
                            <div style="font-size:1.6rem; font-weight:800; color:#111827;">{_dq_pct:.0f}%</div>
                            <div style="font-size:0.78rem; color:#6B7280;">satellite coverage</div>
                            <div style="font-size:0.72rem; color:#9CA3AF; margin-top:2px;">{len(_miss_d)} days missing</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with _r2b:
                _sw = _sp.get("west", 0); _ss = _sp.get("south", 0); _se = _sp.get("east", 0); _sn = _sp.get("north", 0)
                _svg_w, _svg_h = 200, 150
                _rw, _re, _rs, _rn = 76.4, 77.7, 28.0, 29.1
                def _sx(lon): return (lon - _rw) / (_re - _rw) * _svg_w
                def _sy(lat): return _svg_h - (lat - _rs) / (_rn - _rs) * _svg_h
                _bx, _by = _sx(_sw), _sy(_sn)
                _bw_px, _bh_px = _sx(_se) - _sx(_sw), _sy(_ss) - _sy(_sn)
                _ccx, _ccy = _sx(_clon), _sy(_clat)
                _grid = ""
                for _gl in [76.5, 77.0, 77.5]:
                    _gx = _sx(_gl)
                    _grid += f'<line x1="{_gx:.0f}" y1="0" x2="{_gx:.0f}" y2="{_svg_h}" stroke="#E5E7EB" stroke-width="0.5" stroke-dasharray="3,3"/>'
                    _grid += f'<text x="{_gx:.0f}" y="{_svg_h-4}" font-size="7" fill="#9CA3AF" text-anchor="middle">{_gl}°E</text>'
                for _gla in [28.2, 28.5, 28.8]:
                    _gy = _sy(_gla)
                    _grid += f'<line x1="0" y1="{_gy:.0f}" x2="{_svg_w}" y2="{_gy:.0f}" stroke="#E5E7EB" stroke-width="0.5" stroke-dasharray="3,3"/>'
                    _grid += f'<text x="3" y="{_gy-3:.0f}" font-size="7" fill="#9CA3AF">{_gla}°N</text>'
                st.markdown(f"""
                <div class="bento-card">
                    <div class="bento-header">
                        <span class="bento-title">Spatial Coverage</span>
                    </div>
                    <svg width="{_svg_w}" height="{_svg_h}" style="width:100%; height:auto;" viewBox="0 0 {_svg_w} {_svg_h}">
                        <rect width="{_svg_w}" height="{_svg_h}" fill="#FAFBFC" rx="6"/>
                        {_grid}
                        <rect x="{_bx:.1f}" y="{_by:.1f}" width="{_bw_px:.1f}" height="{_bh_px:.1f}" fill="#115E5912" stroke="#115E59" stroke-width="1.5" rx="2" stroke-dasharray="5,2"/>
                        <circle cx="{_ccx:.1f}" cy="{_ccy:.1f}" r="4" fill="#115E59"/>
                        <circle cx="{_ccx:.1f}" cy="{_ccy:.1f}" r="8" fill="none" stroke="#115E59" stroke-width="1" opacity="0.35"/>
                    </svg>
                    <div style="margin-top:8px; font-size:0.78rem; color:#374151;"><b>Center:</b> {_clat:.3f}°N, {_clon:.3f}°E</div>
                    <div style="font-size:0.72rem; color:#9CA3AF; margin-top:2px;">{_sw}°E – {_se}°E · {_ss}°N – {_sn}°N</div>
                </div>
                """, unsafe_allow_html=True)

            with _r2c:
                st.markdown(f"""
                <div class="bento-card">
                    <div class="bento-header">
                        <span class="bento-title">Model Information</span>
                    </div>
                    <div style="font-size:0.82rem;">
                        <div class="list-row">
                            <div class="list-left"><span style="font-weight:700; color:#111827;">Architecture</span></div>
                            <span style="font-size:0.80rem; color:#374151;">CNN + BiLSTM</span>
                        </div>
                        <div class="list-row">
                            <div class="list-left"><span style="font-weight:700; color:#111827;">Sequence</span></div>
                            <span style="font-size:0.80rem; color:#374151;">{_req_d} days</span>
                        </div>
                        <div class="list-row">
                            <div class="list-left"><span style="font-weight:700; color:#111827;">Raw features</span></div>
                            <span style="font-size:0.80rem; color:#374151;">53</span>
                        </div>
                        <div class="list-row">
                            <div class="list-left"><span style="font-weight:700; color:#111827;">Inference tensor</span></div>
                            <span style="font-size:0.80rem; color:#374151;">14 × 84</span>
                        </div>
                        <div class="list-row" style="border-bottom:none;">
                            <div class="list-left"><span style="font-weight:700; color:#111827;">Primary output</span></div>
                            <span style="font-size:0.80rem; color:#115E59; font-weight:600;">Extreme-aware</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ═══════════ 14-DAY SEQUENCE TIMELINE ═══════════
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            _missing_set = set(_miss_d)
            try:
                _seq_start_dt = datetime.strptime(_seq_s, "%Y-%m-%d").date()
                _seq_days = [(_seq_start_dt + timedelta(days=i)) for i in range(_req_d)]
            except Exception:
                _seq_days = []

            if _seq_days:
                _dots = ""
                for _d in _seq_days:
                    _ds = _d.strftime("%Y-%m-%d")
                    _is_m = _ds in _missing_set
                    _dcol = "#EF4444" if _is_m else "#10B981"
                    _dots += f"""
                    <div style="text-align:center; flex:1; min-width:48px;">
                        <div style="width:12px; height:12px; border-radius:50%; background:{_dcol}; margin:0 auto 4px auto; box-shadow:0 0 0 3px {_dcol}20;"></div>
                        <div style="font-size:0.65rem; color:#6B7280; font-weight:600;">{_d.strftime('%d')}</div>
                        <div style="font-size:0.56rem; color:#9CA3AF;">{_d.strftime('%b')}</div>
                    </div>"""
                st.markdown(f"""
                <div class="bento-card">
                    <div class="bento-header">
                        <span class="bento-title">14-Day Sequence Window</span>
                        <span style="font-size:0.74rem; color:#6B7280;">{_seq_s} → {_seq_e}</span>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:2px; margin-top:10px; overflow-x:auto; padding-bottom:4px;">{_dots}</div>
                    <div style="display:flex; gap:16px; margin-top:10px; font-size:0.72rem;">
                        <div style="display:flex; align-items:center; gap:5px;">
                            <div style="width:8px; height:8px; border-radius:50%; background:#10B981;"></div>
                            <span style="color:#6B7280;">Available ({_use_d})</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:5px;">
                            <div style="width:8px; height:8px; border-radius:50%; background:#EF4444;"></div>
                            <span style="color:#6B7280;">Missing ({len(_miss_d)})</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ═══════════ SATELLITE VISUAL & AOD ═══════════
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            _col_sat, _col_aod = st.columns([2, 1])

            with _col_sat:
                _sat_b64 = load_tiff_preview()
                if _sat_b64:
                    st.markdown(f"""
                    <div class="bento-card">
                        <div class="bento-header">
                            <span class="bento-title">Sentinel-5P Visual Reference</span>
                            <span style="font-size:0.66rem; color:#9CA3AF; font-style:italic;">Static reference · Not current prediction raster</span>
                        </div>
                        <img src="data:image/png;base64,{_sat_b64}" style="width:100%; border-radius:10px; margin-top:8px; filter:contrast(1.05) saturate(0.85);" alt="Sentinel-5P HCHO satellite observation reference"/>
                        <div style="font-size:0.70rem; color:#9CA3AF; margin-top:6px;">HCHO tropospheric column density · Sentinel-5P TROPOMI Level-3 · Jan 2025</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bento-card">
                        <div class="bento-header">
                            <span class="bento-title">Satellite Source</span>
                        </div>
                        <div style="font-size:0.90rem; font-weight:700; color:#111827; margin:10px 0 6px 0;">{_sat_src}</div>
                        <div style="font-size:0.78rem; color:#6B7280;">Data provider for atmospheric trace gas observations (CH4, CO, NO2, HCHO) used by the prediction model.</div>
                        <div style="font-size:0.72rem; color:#9CA3AF; margin-top:8px;">Satellite raster preview unavailable (rasterio not installed)</div>
                    </div>
                    """, unsafe_allow_html=True)

            with _col_aod:
                _aod_st = "Used by model" if _aod_used else "Not used by model"
                _aod_ic = "✅" if _aod_used else "○"
                _aod_bg = "#DCFCE7" if _aod_used else "#F9FAFB"
                _aod_bd = "#A7F3D0" if _aod_used else "#E5E7EB"
                _aod_extra = ""
                if _aod_diag:
                    _aod_extra = f'<div style="margin-top:10px; font-size:0.78rem; color:#374151;"><b>Diagnostic:</b> {_aod_diag}</div>'
                else:
                    _aod_extra = '<div style="font-size:0.74rem; color:#9CA3AF; margin-top:6px;">No AOD diagnostic available</div>'
                st.markdown(f"""
                <div class="bento-card">
                    <div class="bento-header">
                        <span class="bento-title">MOSDAC AOD</span>
                    </div>
                    <div style="background:{_aod_bg}; border:1px solid {_aod_bd}; padding:12px; border-radius:10px; margin:8px 0;">
                        <div style="font-size:0.88rem; font-weight:700; color:#111827;">{_aod_ic} {_aod_st}</div>
                        <div style="font-size:0.74rem; color:#6B7280; margin-top:4px;">INSAT-3D Aerosol Optical Depth</div>
                    </div>
                    {_aod_extra}
                </div>
                """, unsafe_allow_html=True)

            # ═══════════ SOURCE / PROVENANCE ═══════════
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-header">
                    <span class="bento-title">Data Provenance</span>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; font-size:0.82rem; margin-top:6px;">
                    <div>
                        <div style="color:#9CA3AF; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Satellite Source</div>
                        <div style="font-weight:700; color:#111827; margin-top:4px;">{_sat_src}</div>
                    </div>
                    <div>
                        <div style="color:#9CA3AF; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Extraction Mode</div>
                        <div style="font-weight:700; color:#111827; margin-top:4px;">training_compat</div>
                    </div>
                    <div>
                        <div style="color:#9CA3AF; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Calibrated</div>
                        <div style="font-weight:700; color:#111827; margin-top:4px;">{'Yes' if _calibrated else 'No'}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ═══════════ COLLAPSIBLE: MODEL DIAGNOSTICS ═══════════
            with st.expander("🔬 Model Diagnostics", expanded=False):
                st.markdown(f"""
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px;">
                    <div class="bento-card" style="border:1px solid #E5E7EB;">
                        <div style="font-size:0.74rem; color:#9CA3AF; font-weight:600;">Raw Model</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#9CA3AF;">{_raw:.2f}</div>
                        <div style="font-size:0.72rem; color:#9CA3AF;">µg/m³</div>
                    </div>
                    <div class="bento-card" style="border:1px solid {_cc}40; border-left:3px solid {_cc};">
                        <div style="font-size:0.74rem; color:#9CA3AF; font-weight:600;">Extreme Model ✦</div>
                        <div style="font-size:1.4rem; font-weight:800; color:{_cc};">{_pm25:.2f}</div>
                        <div style="font-size:0.72rem; color:#115E59; font-weight:600;">µg/m³ · Final prediction</div>
                    </div>
                    <div class="bento-card" style="border:1px solid #E5E7EB;">
                        <div style="font-size:0.74rem; color:#9CA3AF; font-weight:600;">pm25_ug_m3 (ref)</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#9CA3AF;">{_pm25_ref:.2f}</div>
                        <div style="font-size:0.72rem; color:#9CA3AF;">µg/m³ · Not used as final</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if _notes:
                    st.markdown("**Notes:**")
                    for _n in _notes:
                        st.markdown(f"- {_n}")

            # ═══════════ COLLAPSIBLE: DATA QUALITY DETAILS ═══════════
            with st.expander("📊 Data Quality Details", expanded=False):
                if _miss_c:
                    for _day_k, _comps in sorted(_miss_c.items()):
                        try:
                            _dlbl = datetime.strptime(_day_k, "%Y-%m-%d").strftime("%d %b %Y")
                        except Exception:
                            _dlbl = _day_k
                        _cstr = " · ".join(_comps)
                        _full_badge = '<span style="background:#FEE2E2; color:#B91C1C; font-size:0.66rem; font-weight:700; padding:2px 8px; border-radius:10px; margin-left:8px;">Full Miss</span>' if _day_k in _missing_set else ""
                        st.markdown(f"""
                        <div style="padding:8px 0; border-bottom:1px solid #F3F4F6; font-size:0.82rem;">
                            <span style="font-weight:700; color:#111827;">{_dlbl}</span>{_full_badge}
                            <div style="color:#6B7280; margin-top:2px;">{_cstr}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No missing component data available.")

            # ═══════════ PREDICTION PIPELINE VISUAL ═══════════
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="bento-card" style="text-align:center;">
                <div style="font-size:0.76rem; color:#9CA3AF; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Prediction Pipeline</div>
                <div style="display:flex; align-items:center; justify-content:center; gap:0; flex-wrap:wrap;">
                    <div style="background:#F0FDF4; border:1px solid #A7F3D0; padding:8px 14px; border-radius:10px; font-size:0.76rem; font-weight:700; color:#115E59;">🛰️ Sentinel-5P</div>
                    <div style="color:#D1D5DB; padding:0 6px; font-size:1.0rem;">→</div>
                    <div style="background:#F0F9FF; border:1px solid #BAE6FD; padding:8px 14px; border-radius:10px; font-size:0.76rem; font-weight:700; color:#0369A1;">Feature Extraction</div>
                    <div style="color:#D1D5DB; padding:0 6px; font-size:1.0rem;">→</div>
                    <div style="background:#FFF7ED; border:1px solid #FED7AA; padding:8px 14px; border-radius:10px; font-size:0.76rem; font-weight:700; color:#C2410C;">14-Day Sequence</div>
                    <div style="color:#D1D5DB; padding:0 6px; font-size:1.0rem;">→</div>
                    <div style="background:#EEF2FF; border:1px solid #C7D2FE; padding:8px 14px; border-radius:10px; font-size:0.76rem; font-weight:700; color:#4338CA;">🧠 CNN-BiLSTM</div>
                    <div style="color:#D1D5DB; padding:0 6px; font-size:1.0rem;">→</div>
                    <div style="background:#115E59; padding:8px 16px; border-radius:10px; font-size:0.76rem; font-weight:700; color:#FFFFFF;">PM2.5</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif not st.session_state.aqi_pred_error:
            # Initial empty state
            st.markdown("""
            <div style="text-align:center; padding:60px 20px;">
                <div style="font-size:3.2rem; margin-bottom:14px;">🛰️</div>
                <div style="font-size:1.25rem; font-weight:700; color:#111827;">Ready for Prediction</div>
                <div style="font-size:0.88rem; color:#6B7280; margin-top:8px; max-width:520px; margin-left:auto; margin-right:auto; line-height:1.6;">
                    Configure the spatial extent and prediction date above, then click <b>Run Prediction</b> to fetch Sentinel-5P satellite observations and generate an AI-powered PM2.5 prediction for Delhi.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- VIEW 2.2: MULTI-SENSOR SATELLITE EXPLORER ---
    elif app_subpage == "🛰️ Multi-Sensor Satellite Explorer":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">Multi-Sensor Satellite Feature Explorer</h1>
                <p class="page-subtitle">Terrascope openEO Sentinel-5P Level-3 products & MOSDAC INSAT-3D feature distributions.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.markdown("""
            <div class="bento-card">
                <div style="font-weight:700; color:#10B981; font-size:0.82rem;">SENTINEL-5P TROPOMI</div>
                <div style="font-size:1.25rem; font-weight:800; color:#111827; margin:4px 0;">CH4 (Methane)</div>
                <div style="font-size:0.76rem; color:#6B7280;">Level-3 Total Column Density<br>Terrascope openEO Provider</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            st.markdown("""
            <div class="bento-card">
                <div style="font-weight:700; color:#3B82F6; font-size:0.82rem;">SENTINEL-5P TROPOMI</div>
                <div style="font-size:1.25rem; font-weight:800; color:#111827; margin:4px 0;">CO (Carbon Monoxide)</div>
                <div style="font-size:0.76rem; color:#6B7280;">Tropospheric Column Density<br>Incomplete Combustion Tracer</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s3:
            st.markdown("""
            <div class="bento-card">
                <div style="font-weight:700; color:#EF4444; font-size:0.82rem;">SENTINEL-5P TROPOMI</div>
                <div style="font-size:1.25rem; font-weight:800; color:#111827; margin:4px 0;">NO2 (Nitrogen Dioxide)</div>
                <div style="font-size:0.76rem; color:#6B7280;">Tropospheric Vertical Column<br>Vehicular & Thermal Power Proxy</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s4:
            st.markdown("""
            <div class="bento-card">
                <div style="font-weight:700; color:#8B5CF6; font-size:0.82rem;">SENTINEL-5P TROPOMI</div>
                <div style="font-size:1.25rem; font-weight:800; color:#111827; margin:4px 0;">HCHO (Formaldehyde)</div>
                <div style="font-size:0.76rem; color:#6B7280;">Native Resolution Proxy<br>Reactive NMVOC Oxidation</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔬 53-Feature Schema & Statistical Indicators")
        schema_list = sih_metadata.get("features", [])
        if schema_list:
            df_schema = pd.DataFrame({"Feature Name": schema_list, "Group": ["Methane (CH4)" if f.startswith("CH4") else ("Carbon Monoxide (CO)" if f.startswith("CO") else ("Nitrogen Dioxide (NO2)" if f.startswith("NO2") else ("Formaldehyde (HCHO)" if f.startswith("HCHO") else "Spatiotemporal / Calendar"))) for f in schema_list]})
            st.dataframe(df_schema, use_container_width=True, height=350)

    # --- VIEW 2.3: MODEL BENCHMARKS & TAIL STRESS ---
    elif app_subpage == "📊 Model Benchmarks & Tail Stress":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">Model Benchmarks & Extreme Tail Analysis</h1>
                <p class="page-subtitle">Rigorous validation blend search, extreme episode tail metrics, and future regime stress performance.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.markdown("""
            <div class="bento-card-blue">
                <div class="bento-header">
                    <span class="bento-title-light">Representative Test R²</span>
                    <div class="arrow-icon-btn-light">↗</div>
                </div>
                <div class="bento-value-light">0.8899</div>
                <div class="bento-subtext-light">🎯 High Predictive Variance Capture</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b2:
            st.markdown("""
            <div class="bento-card">
                <div class="bento-header">
                    <span class="bento-title">Representative Test MAE</span>
                    <div class="arrow-icon-btn">↗</div>
                </div>
                <div class="bento-value">16.83</div>
                <div class="bento-subtext">µg/m³ Mean Absolute Error (n=1,942)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b3:
            st.markdown("""
            <div class="bento-card">
                <div class="bento-header">
                    <span class="bento-title">Optimal Raw Weight</span>
                    <div class="arrow-icon-btn">↗</div>
                </div>
                <div class="bento-value">0.925</div>
                <div class="bento-subtext" style="color:#1E40AF;">0.075 Extreme Model Injection</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if not df_blend.empty:
            st.markdown("#### 📈 Validation Model Blend Search (MAE & R² vs Raw Model Weight)")
            fig_blend = make_subplots(specs=[[{"secondary_y": True}]])
            fig_blend.add_trace(go.Scatter(x=df_blend['raw_weight'], y=df_blend['MAE'], name="MAE (µg/m³)", line=dict(color="#EF4444", width=3)), secondary_y=False)
            fig_blend.add_trace(go.Scatter(x=df_blend['raw_weight'], y=df_blend['R2'], name="R² Score", line=dict(color="#10B981", width=3)), secondary_y=True)
            
            fig_blend.update_layout(height=350, hovermode="x unified", xaxis_title="Raw Model Blend Weight", margin=dict(l=20, r=20, t=30, b=20))
            fig_blend.update_yaxes(title_text="MAE (µg/m³)", secondary_y=False)
            fig_blend.update_yaxes(title_text="R² Score", secondary_y=True)
            st.plotly_chart(fig_blend, use_container_width=True, config={'displayModeBar': False})

        if not df_rep_tail.empty:
            st.markdown("#### ⚡ Extreme Episode Tail Performance Breakdown")
            st.dataframe(df_rep_tail, use_container_width=True)

    # --- VIEW 2.4: DEEP LEARNING ARCHITECTURE & PIPELINE ---
    elif app_subpage == "🏗️ Deep Learning Architecture & Pipeline":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">CNN-BiLSTM-Attention Dual Architecture</h1>
                <p class="page-subtitle">Technical blueprint of the temporal sequence builder, feature preprocessing, and dual model inference.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="bento-card">
            <h3 style="margin-top:0; color:#1E40AF;">🧠 Deep Learning System Architecture</h3>
            <p>The system ingests 14 consecutive days of satellite measurements across 4 spectral products (CH4, CO, NO2, HCHO), combines them with spatiotemporal positional encodings, applies missing-value masking and robust scaling to produce an <b>84-wide feature tensor</b> over 14 timesteps <code>(Batch, 14, 84)</code>.</p>
            
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; margin-top:14px;">
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:12px;">
                    <div style="font-weight:800; color:#0F172A; font-size:0.95rem;">1. Feature Extraction & Alignment</div>
                    <p style="font-size:0.80rem; color:#475569; margin-top:5px;">
                        • Sentinel-5P Level-3 daily GeoTIFFs (Terrascope openEO)<br>
                        • 10th, 25th, 75th, 90th percentiles + spatial mean/std/min/max<br>
                        • Native-resolution HCHO point & validity fraction
                    </p>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:12px;">
                    <div style="font-weight:800; color:#0F172A; font-size:0.95rem;">2. 1D-CNN Feature Compression</div>
                    <p style="font-size:0.80rem; color:#475569; margin-top:5px;">
                        • 1D Convolutional filter banks<br>
                        • Captures short-term daily transitions & chemical gradients<br>
                        • Batch Normalization & Dropout regularization
                    </p>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:12px;">
                    <div style="font-weight:800; color:#0F172A; font-size:0.95rem;">3. Bidirectional LSTM + Attention</div>
                    <p style="font-size:0.80rem; color:#475569; margin-top:5px;">
                        • Bidirectional temporal recurrence (past & future context)<br>
                        • Self-attention pooling focuses on pollution spike days<br>
                        • Dense regression head with Dual Model Blending
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# DOMAIN 3: CROSS-POLLUTANT ATMOSPHERIC CORRELATION HUB
# ==============================================================================
elif domain_mode == "🌐 Cross-Pollutant Multi-Sensor Hub":

    if app_subpage == "🔗 VOC Precursor & PM2.5 Coupling":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">VOC Precursors & Secondary Aerosol Coupling</h1>
                <p class="page-subtitle">Investigating the chemical link between reactive Formaldehyde (HCHO) oxidation and Secondary Organic Aerosol (SOA / PM2.5) formation.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("""
            <div class="bento-card">
                <h3 style="color:#4C1D95; margin-top:0;">⚡ Photochemical Oxidation Pathway</h3>
                <p style="font-size:0.86rem; color:#374151; line-height:1.5;">
                    Formaldehyde ($HCHO$) is a high-yield oxidation product of non-methane volatile organic compounds ($NMVOCs$).
                    In the presence of sunlight ($h\nu$) and nitrogen oxides ($NO_x$), $HCHO$ photolysis releases hydroperoxyl radicals ($HO_2$), accelerating tropospheric ozone ($O_3$) and secondary organic aerosol ($PM_{2.5}$) synthesis.
                </p>
                <div style="background:#F5F3FF; border:1px solid #DDD6FE; padding:10px; border-radius:10px; font-size:0.80rem; color:#5B21B6;">
                    <b>Key Finding:</b> Clusters with Class A validation (e.g., Sangli-Satara, Bharuch-Surat) exhibit strong downwind enhancement (>1.15x) that directly correlates with downwind ground-level PM2.5 buildup.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            if not df_report.empty:
                fig_corr = px.scatter(
                    df_report,
                    x="final_source_score",
                    y="observed_downwind_upwind_ratio",
                    size="total_FRP",
                    color="validation_class",
                    hover_name="full_location",
                    title="Source Score vs Downwind Plume Enhancement Ratio",
                    color_discrete_map={"Strong": "#10B981", "Moderate": "#F59E0B", "Weak": "#EF4444"},
                    labels={"final_source_score": "Composite Source Score", "observed_downwind_upwind_ratio": "Downwind Ratio (x)"}
                )
                fig_corr.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
                st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})

    elif app_subpage == "🔥 Biomass Fire Radiative Dynamics":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">Biomass Fire Radiative Power (FRP) & Transport</h1>
                <p class="page-subtitle">Analyzing MODIS/VIIRS thermal fire intensity across agricultural stubble corridors and forest zones.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if not df_report.empty:
            fig_frp = px.bar(
                df_report.sort_values(by='total_FRP', ascending=False).head(15),
                x="location_name",
                y="total_FRP",
                color="zone",
                title="Top 15 Hotspots by Active Thermal Fire Radiative Power (MW)",
                labels={"location_name": "Source Region", "total_FRP": "Total Fire FRP (MW)"}
            )
            fig_frp.update_layout(height=400, margin=dict(l=20, r=20, t=35, b=80), xaxis_tickangle=-45)
            st.plotly_chart(fig_frp, use_container_width=True, config={'displayModeBar': False})

    elif app_subpage == "📋 Regulatory Policy & NCAP Action Matrix":
        st.markdown("""
        <div class="page-title-row">
            <div>
                <h1 class="page-main-title">National Clean Air Programme (NCAP) Action Matrix</h1>
                <p class="page-subtitle">Targeted enforcement directives for Central & State Pollution Control Boards (CPCB/SPCB).</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="bento-card">
            <h3 style="margin-top:0; color:#115E59;">🏛️ High-Priority Airshed Intervention Directives</h3>
            <table style="width:100%; border-collapse:collapse; font-size:0.84rem; margin-top:10px;">
                <thead>
                    <tr style="background:#F3F4F6; text-align:left;">
                        <th style="padding:8px; border-bottom:2px solid #E5E7EB;">Airshed / Cluster</th>
                        <th style="padding:8px; border-bottom:2px solid #E5E7EB;">Primary Emission Driver</th>
                        <th style="padding:8px; border-bottom:2px solid #E5E7EB;">Validation Tier</th>
                        <th style="padding:8px; border-bottom:2px solid #E5E7EB;">Recommended Regulatory Action</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;"><b>#34 Sangli-Satara Sugar Belt</b></td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;">Sugarcane agro-waste & bagasse boilers</td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;"><span class="badge-pill-strong">Strong</span></td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;">Mandate wet scrubbers & automated bagasse pelletization.</td>
                    </tr>
                    <tr>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;"><b>#12 Bharuch-Surat Industrial Corridor</b></td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;">Petrochemical & synthetic organic chemicals</td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;"><span class="badge-pill-strong">Strong</span></td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;">Continuous VOC CEMS monitoring & leak detection repairs (LDAR).</td>
                    </tr>
                    <tr>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;"><b>#1 Ludhiana-Jalandhar Corridor</b></td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;">Post-harvest stubble & textile dyeing</td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;"><span class="badge-pill-moderate">Moderate</span></td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;">In-situ CRM machinery deployment & stubble biomass power off-take.</td>
                    </tr>
                    <tr>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;"><b>#16 Bilaspur-Korba Power Basin</b></td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;">Coal power plants & open-cast mining</td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;"><span class="badge-pill-moderate">Moderate</span></td>
                        <td style="padding:8px; border-bottom:1px solid #F3F4F6;">Enforce FGD installation & conveyor dust suppression.</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("AirTrace AI • Unified Atmospheric Remote Sensing & Deep Learning PM2.5 Platform | 2025 Multi-Sensor Satellite System")
