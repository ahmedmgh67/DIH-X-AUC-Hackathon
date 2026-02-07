# FreshFlow - Intelligent Demand Forecasting System

## Overview

FreshFlow is a web-based demand forecasting dashboard for restaurants and grocery stores. It uses LSTM neural networks to predict demand, optimize kitchen prep quantities, and analyze external factors affecting sales.

**Target Use Case:** Deloitte x AUC Hackathon - Fresh Flow Markets (Inventory Management)

## Tech Stack

- **Frontend:** Streamlit (Python)
- **ML Model:** LSTM neural network (PyTorch)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly (interactive charts)
- **Weather API:** Open-Meteo (free, no key required)

## Application Structure

```
freshflow/
├── app.py                    # Main Streamlit entry point
├── pages/
│   ├── 1_forecasting.py      # Demand forecasting drill-down
│   ├── 2_kitchen_prep.py     # Prep quantity calculator
│   └── 3_external_factors.py # Weather/campaign analysis
├── models/
│   ├── lstm_model.py         # LSTM architecture
│   ├── trainer.py            # Training pipeline
│   └── predictor.py          # Inference engine
├── services/
│   ├── data_processor.py     # Data loading & preprocessing
│   ├── weather_service.py    # Open-Meteo API integration
│   └── feature_engineer.py   # External factors encoding
└── utils/
    └── helpers.py            # Shared utilities
```

## Data Flow

1. Load historical orders → Aggregate by day/location/item
2. Fetch weather data → Join with sales data
3. Encode features (time, weather, campaigns)
4. Train LSTM → Generate forecasts
5. Display in dashboard with confidence intervals

## Features

### 1. Executive Summary Page (Landing)

**KPI Cards:**
- Today's Revenue - Actual vs predicted with variance indicator
- Weekly Trend - Week-over-week comparison
- Forecast Accuracy - MAPE from model
- Waste Risk - Items with high overstock probability

**Components:**
- 7-day forecast preview chart
- Top selling items today
- Smart alerts (weather, holidays, campaigns)
- Navigation to detail pages

### 2. Demand Forecasting Page

**Features:**
- Granularity toggle (daily/weekly/monthly)
- Interactive forecast vs actual chart with confidence bands
- Model performance metrics (MAPE, RMSE, R²)
- Item-level forecast breakdown table
- CSV export functionality

### 3. Kitchen Prep Optimizer Page

**Features:**
- Location and date selector
- Adjustable buffer control (5-20%)
- Context banner showing weather/day type/campaigns
- Prep quantity table with confidence indicators
- Historical accuracy tracking
- Smart recommendations based on patterns
- Print/email/export options

**Logic:**
```
Prep Quantity = Forecast × (1 + Buffer%) × Weather Adjustment × Day Adjustment
```

### 4. External Factors Analysis Page

**Features:**
- Day-of-week demand patterns
- Hour-of-day heatmap
- Weather impact correlation
- Campaign performance analytics
- Danish holiday impact calendar
- Compare mode for locations/periods

## LSTM Model Specification

```
Architecture:
- Input Layer: 30-day sequence × 12 features
- LSTM Layer 1: 64 units, dropout 0.2
- LSTM Layer 2: 32 units, dropout 0.2
- Dense Layer: 16 units, ReLU
- Output Layer: 7 units (7-day forecast)

Features (12 total):
- sales_amount, order_count (historical)
- day_of_week (one-hot: 7)
- is_weekend, is_holiday
- temperature, precipitation
- campaign_active

Training:
- Loss: MSE
- Optimizer: Adam, lr=0.001
- Batch size: 32
- Epochs: 100 with early stopping
```

## External Data Sources

### Weather (Open-Meteo API)
- Historical weather for training data
- 7-day forecast for predictions
- Features: temperature, precipitation, weather code
- Rate limit: 10,000 requests/day

### Danish Holidays
- Hardcoded calendar including:
  - Christmas, New Year
  - Easter
  - Constitution Day (June 5)
  - Midsummer
  - Other public holidays

## Data Sources (From Dataset)

| Table | Purpose |
|-------|---------|
| fct_orders | Historical sales data |
| fct_order_items | Item-level sales |
| dim_menu_items | Menu item details |
| dim_places | Restaurant locations |
| fct_campaigns | Campaign history |
| dim_campaigns | Campaign definitions |

## Dependencies

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
plotly>=5.18.0
requests>=2.31.0
scikit-learn>=1.3.0
```

## Success Metrics

- Forecast MAPE < 10%
- Prep accuracy > 90%
- Waste reduction potential demonstrated
- Clear business value presentation
