# ⚡ Smart Energy Forecasting Platform - React Frontend

A modern, responsive React frontend interface for the Electricity Demand Forecaster, built with **Vite**, **Tailwind CSS**, **Recharts**, and **Lucide React**.

---

## 🎨 Theme: Executive Thermal Surge

The application implements the official **Executive Thermal Surge** design system (`THEME.md`):
- **Executive Daylight (Light Mode)**: Sovereign Navy (`#1E3A8A`), Solar Flare Orange (`#EA580C`), Molten Crimson (`#E11D48`).
- **Executive Midnight (Dark Mode)**: Luminous Cyan (`#38BDF8`), Solar Flare Gold (`#FFB703`), Molten Neon Coral (`#FF4D6D`).

---

## 🚀 Quick Start

### 1. Ensure the FastAPI Backend is Running
From the root repository directory:
```bash
# In first terminal:
.venv/Scripts/activate
python server.py
# Server runs at http://localhost:8000
```

### 2. Run the React Frontend
```bash
# In second terminal:
cd frontend
npm run dev
# Frontend runs at http://localhost:5173
```

### 3. Build for Production
```bash
cd frontend
npm run build
```

---

## 🛠️ Key Features & Modules

1. **Top Navigation & Controls Bar**:
   - State selector covering all 36 Indian States and Union Territories (categorized by NR, WR, SR, ER, NER regions).
   - Real-time weather toggle (Open-Meteo Live API vs Climatology Fallback).
   - System Health badge with detailed subsystem inspect modal.
   - Dark / Light mode toggle.

2. **📊 Demand Forecast (Tab 1)**:
   - Horizon selector (7, 14, 30 days).
   - Summary KPIs: Average Daily Demand (MU), Peak Forecast MU & Date, Min Demand, Heatwave Risk Days count, State RMSE ± CI.
   - Interactive Forecast Chart: Demand curve, 90% Confidence Interval band, Peak Demand indicator, CDD background bars, and Mean Temperature overlay.
   - Daily Breakdown Schedule data table with Heatwave and Holiday badges.
   - CSV export capability.

3. **🌡️ Weather Impact Analysis (Tab 2)**:
   - Forecast vs Historical analysis modes.
   - Pearson Correlation KPI Cards: Temperature ↔ Demand & CDD ↔ Demand.
   - AI Thermal Load natural language insight card.
   - Multi-metric dual-axis time series chart (Demand MU vs 7d Baseline vs Temperature vs CDD).
   - ML Feature Importance chart (categorized by Temperature, Humidity, Historical, Calendar, Location, Weather).
   - Seasonal multi-year profile metrics (Summer, Monsoon, Autumn, Winter).

4. **🤖 AI Energy Assistant (Tab 3)**:
   - Grounded RAG conversational interface.
   - Dynamic prompt suggestion chips tailored to selected state.
   - Grounded knowledge citation snippets.
   - Domain scope guardrail indicator.
