@echo off
cd /d "%~dp0"
title Project Data Explorer

python -m streamlit run dashboard_project.py

if %errorlevel% neq 0 pause