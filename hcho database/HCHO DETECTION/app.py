import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from location_metadata import CLUSTER_LOCATIONS, get_location_name
except ImportError:
    from location_metadata import CLUSTER_LOCATIONS, get_location_name

# Set Streamlit page config
st.set_page_config(
    page_title="Donezo • India HCHO Hotspots Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern "Donezo" SaaS Bento-Grid Styling (High Contrast Light Theme)
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

    /* Explicit Dark Typography for all Headings, Paragraphs, and Labels */
    p, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #111827 !important;
    }
    .stCaption, caption, small {
        color: #6B7280 !important;
    }
    
    /* Main App Container padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px;
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
        gap: 6px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: #F9FAFB !important;
        border: 1px solid #F3F4F6 !important;
        padding: 10px 14px !important;
        border-radius: 12px !important;
        transition: all 0.15s ease !important;
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
        font-size: 0.92rem !important;
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
    input[type="text"], input[type="search"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 10px !important;
    }

    /* Slider styling */
    div[data-testid="stSlider"] * {
        color: #1F2937 !important;
    }

    /* Dataframe tables */
    div[data-testid="stDataFrame"] * {
        color: #111827 !important;
    }

    /* Top Navigation Header */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        background: #FFFFFF;
        padding: 12px 24px;
        border-radius: 18px;
        border: 1px solid #ECEEF1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .search-pill {
        display: flex;
        align-items: center;
        background: #F3F4F6;
        padding: 8px 16px;
        border-radius: 12px;
        color: #4B5563 !important;
        font-size: 0.88rem;
        width: 320px;
        gap: 8px;
    }
    .search-pill span {
        color: #4B5563 !important;
    }
    .user-profile-badge {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .user-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #E0E7FF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    
    /* Page Title & Actions */
    .page-title-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 1.5rem;
    }
    .page-main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #111827 !important;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #6B7280 !important;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .btn-solid-green {
        background: #115E59;
        color: #FFFFFF !important;
        padding: 9px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.88rem;
        border: none;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        text-decoration: none;
    }
    .btn-outline {
        background: #FFFFFF;
        color: #1F2937 !important;
        padding: 8px 18px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.88rem;
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
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.02);
        height: 100%;
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .bento-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }
    .bento-card-green {
        background: linear-gradient(135deg, #134E3F 0%, #0D382D 100%);
        border: 1px solid #134E3F;
        border-radius: 20px;
        padding: 22px;
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(19, 78, 63, 0.18);
        height: 100%;
        position: relative;
    }
    .bento-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .bento-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #4B5563 !important;
    }
    .bento-title-light {
        font-size: 0.95rem;
        font-weight: 600;
        color: #E6F4EA !important;
    }
    .arrow-icon-btn {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        color: #374151 !important;
    }
    .arrow-icon-btn-light {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        color: #FFFFFF !important;
    }
    .bento-value {
        font-size: 2.3rem;
        font-weight: 800;
        color: #111827 !important;
        line-height: 1.1;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .bento-value-light {
        font-size: 2.3rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.1;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .bento-subtext {
        font-size: 0.82rem;
        color: #16A34A !important;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .bento-subtext-light {
        font-size: 0.82rem;
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
        padding: 11px 0;
        border-bottom: 1px solid #F3F4F6;
    }
    .list-row:last-child {
        border-bottom: none;
    }
    .list-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .list-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .list-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #1F2937 !important;
        margin-bottom: 2px;
    }
    .list-subtitle {
        font-size: 0.76rem;
        color: #9CA3AF !important;
    }
    .badge-pill-strong {
        background: #DCFCE7 !important;
        color: #15803D !important;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
    }
    .badge-pill-moderate {
        background: #FEF3C7 !important;
        color: #B45309 !important;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
    }
    .badge-pill-weak {
        background: #FEE2E2 !important;
        color: #B91C1C !important;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
    }

    /* Dark Green Tracker Card */
    .tracker-card {
        background: linear-gradient(145deg, #093327 0%, #031D16 100%);
        border-radius: 20px;
        padding: 22px;
        color: #FFFFFF !important;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .tracker-digits {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin: 12px 0;
        color: #FFFFFF !important;
    }
    .tracker-card * {
        color: #FFFFFF !important;
    }
    .tracker-btn-row {
        display: flex;
        gap: 12px;
        margin-top: 14px;
    }
    .tracker-btn-white {
        background: #FFFFFF !important;
        color: #031D16 !important;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        border: none;
    }
    .tracker-btn-red {
        background: #EF4444 !important;
        color: #FFFFFF !important;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        border: none;
    }

    /* Sidebar Header and Promo */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.35rem;
        font-weight: 800;
        color: #111827 !important;
        margin-bottom: 24px;
    }
    .sidebar-brand span {
        color: #111827 !important;
    }
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #6B7280 !important;
        text-transform: uppercase;
        margin-top: 20px;
        margin-bottom: 8px;
    }
    .sidebar-promo-box {
        background: linear-gradient(135deg, #0E3E32 0%, #06231B 100%) !important;
        border-radius: 16px;
        padding: 18px;
        color: #FFFFFF !important;
        margin-top: 30px;
    }
    .sidebar-promo-box * {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# DOMESTIC INDIAN LOCAL HOTSPOT REGIONS (28 CLUSTERS)
# ----------------------------------------------------
LOCAL_INDIAN_REGIONS = {cid: info for cid, info in CLUSTER_LOCATIONS.items() if info.get('country') == 'India'}


# ----------------------------------------------------
# DATA LOADING & FILTERING TO LOCAL INDIAN REGIONS
# ----------------------------------------------------
@st.cache_data
def load_india_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(base_dir, "features")
    if not os.path.exists(features_dir):
        features_dir = os.path.join(base_dir, "HCHO DETECTION", "features")
    
    report_file = os.path.join(features_dir, "FINAL_SOURCE_REGION_REPORT_2025.csv")
    monthly_file = os.path.join(features_dir, "SOURCE_RECEPTOR_TRANSPORT_MONTHLY_2025.csv")
    hdbscan_pts_file = os.path.join(features_dir, "HCHO_SOURCE_CLUSTERS_HDBSCAN_2025.csv")
    validation_file = os.path.join(features_dir, "FINAL_SOURCE_REGION_VALIDATION_2025.csv")
    summary_file = os.path.join(features_dir, "FINAL_SOURCE_REGION_SUMMARY_2025.csv")
    
    df_report = pd.read_csv(report_file) if os.path.exists(report_file) else pd.DataFrame()
    df_monthly = pd.read_csv(monthly_file) if os.path.exists(monthly_file) else pd.DataFrame()
    df_points = pd.read_csv(hdbscan_pts_file) if os.path.exists(hdbscan_pts_file) else pd.DataFrame()
    df_validation = pd.read_csv(validation_file) if os.path.exists(validation_file) else pd.DataFrame()
    df_summary = pd.read_csv(summary_file) if os.path.exists(summary_file) else pd.DataFrame()
    
    # FILTER EXCLUSIVELY TO INDIAN LOCAL REGIONS
    indian_cluster_ids = list(LOCAL_INDIAN_REGIONS.keys())
    
    if not df_report.empty:
        df_report['cluster_id'] = df_report['cluster_id'].astype(int)
        df_report = df_report[df_report['cluster_id'].isin(indian_cluster_ids)].copy()
        
        # Enrich metadata
        df_report['location_name'] = df_report['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['location_name'])
        df_report['state'] = df_report['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['state'])
        df_report['zone'] = df_report['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['zone'])
        df_report['district'] = df_report['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['district'])
        df_report['region_type'] = df_report['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['region_type'])
        df_report['description'] = df_report['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['description'])
        df_report['full_location'] = df_report['location_name'] + ", " + df_report['state']
        df_report['map_label'] = "#" + df_report['validation_rank'].astype(str) + " " + df_report['location_name']
        
    if not df_monthly.empty:
        df_monthly['cluster_id'] = df_monthly['cluster_id'].astype(int)
        df_monthly = df_monthly[df_monthly['cluster_id'].isin(indian_cluster_ids)].copy()
        df_monthly['location_name'] = df_monthly['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['location_name'])
        df_monthly['state'] = df_monthly['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['state'])
        df_monthly['zone'] = df_monthly['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['zone'])
        df_monthly['cluster_label'] = "#" + df_monthly['cluster_id'].astype(str) + " " + df_monthly['location_name']

    if not df_points.empty:
        df_points['cluster_id'] = df_points['cluster_id'].astype(int)
        df_points = df_points[df_points['cluster_id'].isin(indian_cluster_ids)].copy()
        df_points['location_name'] = df_points['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['location_name'])
        df_points['state'] = df_points['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['state'])
        df_points['zone'] = df_points['cluster_id'].apply(lambda cid: LOCAL_INDIAN_REGIONS[cid]['zone'])

    return df_report, df_monthly, df_points, df_validation, df_summary, features_dir

df_report, df_monthly, df_points, df_validation, df_summary, features_dir = load_india_data()

# ----------------------------------------------------
# SIDEBAR NAVIGATION & LOCAL FILTERS
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="background: #115E59; color: white; width: 34px; height: 34px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; font-size: 1.1rem;">🌿</span>
        <span>India HCHO Trace</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">MENU</div>', unsafe_allow_html=True)
    
    app_mode = st.radio(
        "Navigation",
        options=[
            "📊 Dashboard Overview",
            "🗺️ Local Hotspot Map",
            "📈 Transport & Dynamics",
            "🔬 Grid Points & Clusters",
            "📑 Report & Downloads",
            "🖼️ Research Figures"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="sidebar-section-title">DOMESTIC FILTERS</div>', unsafe_allow_html=True)
    
    # State Filter
    all_states = sorted(df_report['state'].dropna().unique().tolist()) if 'state' in df_report else []
    selected_states = st.multiselect("Select State / Region:", options=all_states, default=all_states)
    
    # Zone Filter
    all_zones = sorted(df_report['zone'].dropna().unique().tolist()) if 'zone' in df_report else []
    selected_zones = st.multiselect("Geographic Zone:", options=all_zones, default=all_zones)

    # Validation Class Filter
    val_classes = sorted(df_report['validation_class'].dropna().unique().tolist()) if 'validation_class' in df_report else []
    selected_classes = st.multiselect("Validation Confidence:", options=val_classes, default=val_classes)

    st.markdown("""
    <div class="sidebar-promo-box">
        <div style="font-size: 0.82rem; font-weight: 700; color: #86EFAC; margin-bottom: 4px;">INDIA OBSERVATORY</div>
        <div style="font-weight: 700; font-size: 0.95rem; line-height: 1.25; margin-bottom: 8px;">28 Local Source Regions</div>
        <div style="font-size: 0.78rem; opacity: 0.8; margin-bottom: 12px;">Maharashtra • Gujarat • MP • Punjab • TN • KA • UP</div>
        <a href="#downloads" style="background: #10B981; color: white; padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 700; text-decoration: none; display: inline-block;">Export Data</a>
    </div>
    """, unsafe_allow_html=True)

# Apply global sidebar filters
filtered_report = df_report.copy()
if selected_states:
    filtered_report = filtered_report[filtered_report['state'].isin(selected_states)]
if selected_zones:
    filtered_report = filtered_report[filtered_report['zone'].isin(selected_zones)]
if selected_classes:
    filtered_report = filtered_report[filtered_report['validation_class'].isin(selected_classes)]

# ----------------------------------------------------
# TOP NAVIGATION BAR
# ----------------------------------------------------
st.markdown("""
<div class="top-navbar">
    <div class="search-pill">
        <span>🔍</span>
        <span>Search Indian states, districts, industrial hubs...</span>
        <span style="margin-left: auto; background: #E5E7EB; padding: 2px 6px; border-radius: 6px; font-size: 0.7rem; font-weight: 700;">⌘ F</span>
    </div>
    <div class="user-profile-badge">
        <span style="font-size: 1.1rem; cursor: pointer; margin-right: 6px;">🔔</span>
        <span style="font-size: 1.1rem; cursor: pointer; margin-right: 12px;">✉️</span>
        <div class="user-avatar">🇮🇳</div>
        <div>
            <div style="font-weight: 700; font-size: 0.85rem; color: #111827; line-height: 1.1;">India Domestic Observatory</div>
            <div style="font-size: 0.75rem; color: #6B7280;">28 Hotspots • TROPOMI Satellite 2025</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# VIEW 1: DASHBOARD OVERVIEW (BENTO GRID DESIGN)
# ----------------------------------------------------
if app_mode == "📊 Dashboard Overview":
    
    # Page Title Row
    st.markdown("""
    <div class="page-title-row">
        <div>
            <h1 class="page-main-title">India HCHO Hotspots Dashboard</h1>
            <p class="page-subtitle">Domestic monitoring of formaldehyde hotspots, agricultural crop burning, and industrial chemical corridors (2025).</p>
        </div>
        <div style="display: flex; gap: 10px;">
            <a class="btn-solid-green" href="javascript:void(0);">＋ Filter State</a>
            <a class="btn-outline" href="javascript:void(0);">Export Report</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ------------------------------------------------
    # ROW 1: 4 BENTO KPI CARDS
    # ------------------------------------------------
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
    
    # ------------------------------------------------
    # ROW 2: BENTO MIDDLE ROW
    # ------------------------------------------------
    mid_col1, mid_col2, mid_col3 = st.columns([1.4, 1.2, 1.4])
    
    with mid_col1:
        st.markdown("""
        <div class="bento-card">
            <div style="font-weight: 800; font-size: 1.05rem; color: #111827; margin-bottom: 12px;">Domestic Emission Analytics (Monthly Fire FRP)</div>
        """, unsafe_allow_html=True)
        
        # Monthly bar chart for Indian regions
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
                height=210,
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
                <div style="font-weight: 800; font-size: 1.05rem; color: #111827; margin-bottom: 8px;">Top Local Hotspot</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #115E59; margin-bottom: 4px;">#{top_hotspot.get('validation_rank', 1)} {top_hotspot.get('location_name', 'Sangli-Satara')}</div>
                <div style="font-size: 0.82rem; color: #6B7280; margin-bottom: 8px;">📍 {top_hotspot.get('state', 'Maharashtra')} ({top_hotspot.get('zone', 'West Zone')})</div>
                <div style="font-size: 0.78rem; color: #4B5563; background:#F0FDF4; padding:8px 12px; border-radius:10px; border:1px solid #DCFCE7;">
                    ⚡ <b>Downwind Ratio:</b> {top_hotspot.get('observed_downwind_upwind_ratio', 1.17):.3f}x<br>
                    🔥 <b>Total Fire FRP:</b> {top_hotspot.get('total_FRP', 0):,.0f} MW<br>
                    🏭 <b>Type:</b> {top_hotspot.get('region_type', 'Agro-Industrial')}
                </div>
            </div>
            <div style="margin-top: 14px;">
                <button style="width: 100%; background: #134E3F; color: white; border: none; padding: 10px; border-radius: 12px; font-weight: 700; font-size: 0.85rem; cursor: pointer;">
                    🗺️ Open Local Hotspots Map
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mid_col3:
        st.markdown("""
        <div class="bento-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 800; font-size: 1.05rem; color: #111827;">Top Ranked Indian Hotspots</span>
                <span style="background: #F3F4F6; color: #374151; padding: 3px 8px; border-radius: 8px; font-size: 0.72rem; font-weight: 700;">Ranked</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Display top 4 local hotspots in list style
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
    
    # ------------------------------------------------
    # ROW 3: BENTO BOTTOM ROW
    # ------------------------------------------------
    bot_col1, bot_col2, bot_col3 = st.columns([1.4, 1.2, 1.4])
    
    with bot_col1:
        st.markdown("""
        <div class="bento-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 800; font-size: 1.05rem; color: #111827;">Hotspot Count by State</span>
                <span style="background: #F3F4F6; color: #374151; padding: 3px 8px; border-radius: 8px; font-size: 0.72rem; font-weight: 700;">States</span>
            </div>
        """, unsafe_allow_html=True)
        
        state_counts = filtered_report['state'].value_counts()
        for sname, count in state_counts.head(5).items():
            st.markdown(f"""
            <div class="list-row">
                <div class="list-left">
                    <span style="font-size: 1.1rem;">🏛️</span>
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
            <div style="font-weight: 800; font-size: 1.05rem; color: #111827; margin-bottom: 6px;">Domestic Validation Progress</div>
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
            height=180,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            annotations=[dict(text=f"<b>{validated_pct:.0f}%</b><br><span style='font-size:11px;color:#9CA3AF;'>Validated</span>", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 12px; font-size: 0.72rem; font-weight: 700; color: #6B7280;">
            <span><span style="color:#134E3F;">●</span> Strong</span>
            <span><span style="color:#34D399;">●</span> Moderate</span>
            <span><span style="color:#D1D5DB;">●</span> Weak</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with bot_col3:
        total_cells_india = int(filtered_report['cells'].sum()) if 'cells' in filtered_report else 0
        st.markdown(f"""
        <div class="tracker-card">
            <div style="font-weight: 700; font-size: 0.88rem; color: #86EFAC;">Domestic Monitoring Sync</div>
            <div class="tracker-digits">{total_cells_india:,}</div>
            <div style="font-size: 0.8rem; opacity: 0.85;">High-Density Spatial Grid Cells Monitored across India</div>
            <div class="tracker-btn-row">
                <button class="tracker-btn-white">⏸</button>
                <button class="tracker-btn-red">⏹</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# VIEW 2: INTERACTIVE LOCAL HOTSPOT MAP
# ----------------------------------------------------
elif app_mode == "🗺️ Local Hotspot Map":
    st.markdown("""
    <div class="page-title-row">
        <div>
            <h1 class="page-main-title">Interactive Local Hotspots Map</h1>
            <p class="page-subtitle">Explore 28 domestic Indian source regions across agricultural zones, industrial corridors, and coal basins.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Map Controls Card
    with st.container():
        col_preset, col_view_opt, col_label_opt = st.columns([2, 1, 1])
        
        preset_options = [
            "🇮🇳 All India Overview",
            "🌾 Northern Region (Punjab, UP, Rajasthan)",
            "🏭 Western Region (Maharashtra & Gujarat)",
            "⚡ Central Energy Hub (MP, Chhattisgarh)",
            "🌴 Southern Peninsula (TN, KA, AP, Kerala)",
            "🌲 Eastern & North-Eastern (Bihar, WB, Jharkhand, Assam)"
        ]
        
        cluster_targets = []
        for _, r in df_report.sort_values(by='validation_rank').iterrows():
            cluster_targets.append(f"🎯 #{int(r['validation_rank'])} {r['location_name']} ({r['state']})")
        
        with col_preset:
            selected_focus = st.selectbox("📍 Quick Zoom / Fly-To Region:", options=preset_options + ["--- Local Hotspots ---"] + cluster_targets, index=0)
            
        with col_view_opt:
            map_style = st.selectbox("🗺️ Base Map Layer:", options=["open-street-map", "carto-positron", "carto-darkmatter", "white-bg"], index=0)
            
        with col_label_opt:
            show_labels = st.checkbox("🏷️ Show District/City Labels on Map", value=True)
            
        col_map_opt1, col_map_opt2 = st.columns(2)
        with col_map_opt1:
            color_metric = st.selectbox("Color Markers By:", options=["validation_class", "zone", "final_source_score", "total_FRP", "total_fire_count", "observed_downwind_upwind_ratio", "mean_wind_speed"], format_func=lambda x: x.replace('_', ' ').title())
        with col_map_opt2:
            size_metric = st.selectbox("Size Markers By:", options=["cells", "total_FRP", "total_fire_count", "final_source_score"], format_func=lambda x: x.replace('_', ' ').title())

    # Determine center and zoom level
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
            size_max=32,
            zoom=zoom_level,
            center={"lat": center_lat, "lon": center_lon},
            mapbox_style=map_style,
            color_discrete_map=color_discrete_map if color_metric == "validation_class" else None,
            color_continuous_scale="Plasma" if color_metric not in ["validation_class", "zone", "state"] else None,
            text="map_label" if show_labels else None,
            hover_name="full_location",
            hover_data=hover_data,
            height=660
        )
        
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig_config = {"scrollZoom": True, "displayModeBar": True, "displaylogo": False, "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"]}
        st.plotly_chart(fig_map, use_container_width=True, config=fig_config)
        
        # Selected Location Inspector Box
        st.markdown("#### 📍 Selected Hotspot Profile")
        inspect_cid = selected_cid_detail if selected_cid_detail is not None else int(filtered_report.iloc[0]['cluster_id'])
        chosen_info = df_report[df_report['cluster_id'] == inspect_cid].iloc[0]
        val_color = "#10B981" if chosen_info['validation_class'] == "Strong" else ("#F59E0B" if chosen_info['validation_class'] == "Moderate" else "#EF4444")
        
        st.markdown(f"""
        <div class="bento-card" style="background:#F0FDF4; border: 1px solid #DCFCE7;">
            <h3 style="margin-top:0; color:#115E59;">📍 {chosen_info['location_name']}, {chosen_info['state']}</h3>
            <p style="color:#374151;"><b>Key Districts / Areas:</b> {chosen_info['district']} • <b>Zone:</b> {chosen_info['zone']}</p>
            <p style="color:#374151;"><b>Landscape & Primary Emission Driver:</b> {chosen_info['description']}</p>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 10px;">
                <div><b>Validation Rank:</b> #{chosen_info['validation_rank']} in India</div>
                <div><b>Confidence Level:</b> <span style="color:{val_color}; font-weight:700;">{chosen_info['validation_class']}</span></div>
                <div><b>Coordinates:</b> {chosen_info['mean_latitude']:.3f}° N, {chosen_info['mean_longitude']:.3f}° E</div>
                <div><b>Spatial Cells:</b> {int(chosen_info['cells']):,}</div>
                <div><b>Total Fire Count:</b> {int(chosen_info['total_fire_count']):,} fires</div>
                <div><b>Total Fire FRP:</b> {chosen_info['total_FRP']:,.1f} MW</div>
                <div><b>Downwind Enhancement:</b> {chosen_info['observed_downwind_upwind_ratio']:.3f}x ratio</div>
                <div><b>Mean Wind Speed:</b> {chosen_info['mean_wind_speed']:.2f} m/s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No hotspots match current filters.")

# ----------------------------------------------------
# VIEW 3: TRANSPORT & DYNAMICS
# ----------------------------------------------------
elif app_mode == "📈 Transport & Dynamics":
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
            fig_monthly.add_trace(go.Scatter(x=cluster_monthly['month'], y=cluster_monthly['mean_upwind_HCHO'], name="Upwind HCHO (Baseline)", line=dict(color="#10B981", width=2, dash="dash")), row=1, col=1)
            fig_monthly.add_trace(go.Bar(x=cluster_monthly['month'], y=cluster_monthly['source_FRP'], name="Fire FRP (MW)", marker_color="#115E59", opacity=0.7), row=2, col=1, secondary_y=False)
            fig_monthly.add_trace(go.Scatter(x=cluster_monthly['month'], y=cluster_monthly['mean_wind_speed'], name="Wind Speed (m/s)", line=dict(color="#06B6D4", width=2.5)), row=2, col=1, secondary_y=True)
            
            fig_monthly.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified")
            st.plotly_chart(fig_monthly, use_container_width=True)

# ----------------------------------------------------
# VIEW 4: GRID POINTS & CLUSTERS
# ----------------------------------------------------
elif app_mode == "🔬 Grid Points & Clusters":
    st.markdown("""
    <div class="page-title-row">
        <div>
            <h1 class="page-main-title">Grid Points & HDBSCAN Clusters</h1>
            <p class="page-subtitle">Candidate grid cells across the Indian subcontinent clustered via HDBSCAN.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not df_points.empty:
        col_g1, col_g2 = st.columns([1, 3])
        with col_g1:
            sample_rate = st.slider("Sample fraction (%):", min_value=10, max_value=100, value=50, step=10)
            sampled_df = df_points.sample(frac=sample_rate/100.0, random_state=42).copy()
            color_choice = st.selectbox("Color Cells By:", options=["state", "zone", "location_name", "mean_HCHO", "total_FRP", "total_fire_count", "cluster_probability"])
            
        with col_g2:
            if color_choice in ["state", "zone", "location_name"]:
                fig_cells = px.scatter(sampled_df, x="longitude", y="latitude", color=color_choice, title=f"Spatial Point Cloud ({len(sampled_df):,} sampled Indian cells)", height=500)
            else:
                fig_cells = px.scatter(sampled_df, x="longitude", y="latitude", color=color_choice, color_continuous_scale="Turbo", title=f"Spatial Point Cloud ({len(sampled_df):,} sampled Indian cells)", height=500)
            fig_cells.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cells, use_container_width=True)

# ----------------------------------------------------
# VIEW 5: REPORT & DOWNLOADS
# ----------------------------------------------------
elif app_mode == "📑 Report & Downloads":
    st.markdown("""
    <div class="page-title-row">
        <div>
            <h1 class="page-main-title">India Domestic Report & Data Export</h1>
            <p class="page-subtitle">Complete table of 28 validated Indian formaldehyde source regions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    search_query = st.text_input("🔍 Search Table (by state, district, or region name):", "")
    display_df = filtered_report.copy()
    if search_query:
        display_df = display_df[display_df['location_name'].astype(str).str.contains(search_query, case=False) | display_df['state'].astype(str).str.contains(search_query, case=False) | display_df['district'].astype(str).str.contains(search_query, case=False)]
        
    st.dataframe(display_df, use_container_width=True, height=450)
    
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download India Hotspots CSV", data=csv_bytes, file_name="India_HCHO_Hotspots_2025.csv", mime="text/csv")

# ----------------------------------------------------
# VIEW 6: RESEARCH FIGURES (SINGLE FIGURE VIEWER)
# ----------------------------------------------------
elif app_mode == "🖼️ Research Figures":
    st.markdown("""
    <div class="page-title-row">
        <div>
            <h1 class="page-main-title">Publication Figures</h1>
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
    fig_path = os.path.join(features_dir, current_fig["filename"])
    
    st.markdown(f"### {current_fig['title']}")
    st.caption(f"{current_fig['desc']} • File: `{current_fig['filename']}`")
    
    if os.path.exists(fig_path):
        st.image(fig_path, use_column_width=True)
        with open(fig_path, "rb") as file:
            st.download_button(label=f"💾 Download High-Res Image ({current_fig['filename']})", data=file, file_name=current_fig["filename"], mime="image/png")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.caption("Donezo • India Domestic Formaldehyde (HCHO) Satellite & Wind Transport Dashboard | 2025 Multi-Sensor Analysis")
