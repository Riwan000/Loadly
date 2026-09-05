# ⚡ Smart Energy Forecasting Platform — Design System & Theme Guide

This document defines the official color palettes and styling specifications for the **Smart Energy Forecasting Platform**, featuring the custom hybrid theme: **Executive Thermal Surge** (Fusion of *Grid Executive* and *Solar Heatwave & CDD Spike*).

---

## 🌟 Primary Theme: Executive Thermal Surge

A balanced design system that bridges the **authoritative structure & institutional precision of Grid Executive** (Deep Navy / Slate) with the **high-impact thermal dynamics of Solar Heatwave & CDD Spikes** (Solar Flare Gold / Molten Crimson).

---

### ☀️ Edition A: Executive Daylight (Light Mode)

> **Best for:** Executive briefings, regulatory reporting, presentation decks, and daylight operations.

#### 1. Color Palette Tokens

| Role | Color Name | Hex Code | Opacity / Notes |
| :--- | :--- | :--- | :--- |
| **Primary Accent / Forecast Curve** | Sovereign Navy | `#1E3A8A` | High contrast, grounds the demand prediction |
| **Confidence Interval (90% CI)** | Navy Mist Fill | `rgba(30, 58, 138, 0.12)` | Clean, non-distracting uncertainty band |
| **Weather / Ambient Temperature** | Solar Flare Orange | `#EA580C` | Clear separation from demand curve |
| **Peak Surge / CDD Heatwave Alert**| Molten Crimson | `#E11D48` | Severe heatwave & cooling degree spikes |
| **CDD Bar Chart Fill** | Crimson Wash | `rgba(225, 29, 72, 0.28)` | Background bar indicator for CDD surges |
| **Historical Baseline (7d Avg)** | Steel Slate (Dashed) | `#64748B` | Reference baseline without weather |
| **Main Background** | Executive Slate | `#F8FAFC` | Clean, glare-free off-white surface |
| **Card & Sidebar Background** | Pure Snow White | `#FFFFFF` | Elevated containers and interactive sidebars |
| **Card / Container Borders** | Muted Slate Border | `#E2E8F0` | Subtle element separation |
| **Primary Typography** | Deep Obsidian Ink | `#0F172A` | High-readability header & body text |
| **Secondary / Muted Typography** | Cool Slate Text | `#475569` | Captions, units, and timestamps |

#### 2. `.streamlit/config.toml` Configuration

```toml
[theme]
primaryColor = "#1E3A8A"
backgroundColor = "#F8FAFC"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#0F172A"
font = "sans serif"
```

---

### 🌙 Edition B: Executive Midnight (Dark Mode)

> **Best for:** SCADA control rooms, continuous monitoring dashboards, energy trading desks, and low-light environments.

#### 1. Color Palette Tokens

| Role | Color Name | Hex Code | Opacity / Notes |
| :--- | :--- | :--- | :--- |
| **Primary Accent / Forecast Curve** | Luminous Marine Cyan | `#38BDF8` | High-visibility primary demand curve |
| **Confidence Interval (90% CI)** | Cyan Glow Fill | `rgba(56, 189, 248, 0.18)` | Soft glowing uncertainty envelope |
| **Weather / Ambient Temperature** | Solar Flare Gold | `#FFB703` | Warm sunlight overlay trace |
| **Peak Surge / CDD Heatwave Alert**| Molten Neon Coral | `#FF4D6D` | High-urgency alert markers & CDD spikes |
| **CDD Bar Chart Fill** | Molten Coral Wash | `rgba(255, 77, 109, 0.32)`| Background thermal load indicator |
| **Historical Baseline (7d Avg)** | Slate Gray (Dashed) | `#64748B` | Reference baseline curve |
| **Main Background** | Executive Navy Abyss | `#0A0F1D` | Ultra-dark, fatigue-reducing background |
| **Card & Sidebar Background** | Executive Indigo Slate | `#131C31` | Elevated dark surfaces |
| **Card / Container Borders** | Indigo Gridline | `#233050` | Crisp container outlines |
| **Primary Typography** | Crisp Ice White | `#F8FAFC` | Crisp, high-contrast text |
| **Secondary / Muted Typography** | Slate Blue Text | `#94A3B8` | Subtitles, labels, and legends |

#### 2. `.streamlit/config.toml` Configuration

```toml
[theme]
primaryColor = "#38BDF8"
backgroundColor = "#0A0F1D"
secondaryBackgroundColor = "#131C31"
textColor = "#F8FAFC"
font = "sans serif"
```

---

## 📊 Plotly Chart Color Implementation

To apply the **Executive Daylight** or **Executive Midnight** styling to Plotly charts in `visualization.py` and `ui/forecast_tab.py`:

```python
# Theme Configuration Dictionary
THEME_CONFIG = {
    "light": {
        "plotly_template": "plotly_white",
        "demand_line": "#1E3A8A",
        "ci_fill": "rgba(30, 58, 138, 0.12)",
        "temp_mean": "#EA580C",
        "temp_max": "#C2410C",
        "extreme_heat": "#E11D48",
        "cdd_bar": "rgba(225, 29, 72, 0.28)",
        "baseline": "#64748B",
        "peak_marker_fill": "#1E3A8A",
        "peak_marker_stroke": "#E11D48",
    },
    "dark": {
        "plotly_template": "plotly_dark",
        "demand_line": "#38BDF8",
        "ci_fill": "rgba(56, 189, 248, 0.18)",
        "temp_mean": "#FFB703",
        "temp_max": "#FB923C",
        "extreme_heat": "#FF4D6D",
        "cdd_bar": "rgba(255, 77, 109, 0.32)",
        "baseline": "#64748B",
        "peak_marker_fill": "#38BDF8",
        "peak_marker_stroke": "#FF4D6D",
    }
}
```

### Feature Importance Category Color Mapping

```python
FEATURE_CATEGORY_COLORS = {
    "Temperature": "#EA580C",    # Solar Flare Orange
    "Humidity": "#0D9488",       # Teal Green (UTCI / Dewpoint)
    "Calendar": "#7C3AED",       # Purple (Holidays, Seasons)
    "Weather": "#F59E0B",        # Amber (Solar Radiation, Wind)
    "Historical": "#1E3A8A",     # Deep Navy (Rolling Averages, Lag)
    "Location": "#64748B",       # Slate Gray (State / Region)
    "Other": "#94A3B8"
}
```

---

## 🎨 Alternative Explored Themes (Archive)

For future reference, here are the standalone themes developed during exploration:

### 1. 🌱 CleanTech Grid (Modern Renewable Light)
* **Primary (Demand):** `#0F766E` (Deep Teal)
* **Weather (Temp):** `#D97706` (Solar Ochre)
* **Alert:** `#DC2626` (Red)
* **Background:** `#F8FAFC` | **Card:** `#FFFFFF`

### 2. ⚡ Cyber Grid (Mission-Control Dark)
* **Primary (Demand):** `#00E5FF` (Electric Cyan)
* **Weather (Temp):** `#FFB703` (Solar Gold)
* **Alert:** `#FF3B5C` (Neon Coral)
* **Background:** `#0B0F19` | **Card:** `#131B2E`

### 3. 🏢 Industrial Slate (Utility SCADA Dark)
* **Primary (Demand):** `#2563EB` (Cobalt Steel)
* **Weather (Temp):** `#EAB308` (Warning Amber)
* **Alert:** `#EF4444` (Vermillion)
* **Background:** `#18181B` | **Card:** `#27272A`

### 4. 🌡️ Thermal Horizon (Weather-First Twilight)
* **Primary (Demand):** `#38BDF8` (Ice Blue)
* **Weather (Temp):** `#FB923C` (Sunset Tangerine)
* **Alert:** `#E11D48` (Crimson Flare)
* **Background:** `#0F172A` | **Card:** `#1E293B`
