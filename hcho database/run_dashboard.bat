@echo off
title AirTrace AI - Unified Dashboard Launcher
echo ========================================================
echo Starting Unified Atmospheric & PM2.5 Dashboard...
echo ========================================================
cd /d "%~dp0\.."
streamlit run unified_dashboard.py
pause
