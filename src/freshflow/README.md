# FreshFlow - Intelligent Demand Forecasting

A web-based demand forecasting dashboard for restaurants and grocery stores, built for the Deloitte x AUC Hackathon.

## Features

### 1. Executive Dashboard
- Real-time KPIs: Revenue, trends, forecast accuracy
- 7-day demand forecast with confidence intervals
- Smart alerts for weather, weekends, and campaigns
- Top selling items overview

### 2. Demand Forecasting
- LSTM-powered daily/weekly/monthly predictions
- Item-level demand breakdown
- Model performance metrics (MAPE, RMSE, R²)
- CSV export functionality

### 3. Kitchen Prep Optimizer
- Calculate optimal prep quantities
- Configurable safety buffer (5-25%)
- Weather and day-type adjustments
- Printable prep sheets

### 4. External Factors Analysis
- Day-of-week demand patterns
- Weather impact correlation
- Campaign performance tracking
- Danish holiday impact calendar

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
cd src/freshflow
streamlit run app.py
```

Or use the run script:
```bash
./run_freshflow.sh
```

## Architecture

```
freshflow/
├── app.py                    # Main dashboard (Executive Summary)
├── pages/
│   ├── 1_forecasting.py      # Demand forecasting details
│   ├── 2_kitchen_prep.py     # Kitchen prep optimizer
│   └── 3_external_factors.py # External factors analysis
├── models/
│   ├── lstm_model.py         # LSTM neural network
│   ├── trainer.py            # Training pipeline
│   └── predictor.py          # Forecast generation
├── services/
│   ├── data_processor.py     # Data loading & preprocessing
│   ├── weather_service.py    # Open-Meteo API integration
│   └── feature_engineer.py   # Feature creation
└── utils/
    └── helpers.py            # Utility functions
```

## Technology Stack

- **Frontend**: Streamlit
- **ML Model**: PyTorch LSTM
- **Visualization**: Plotly
- **Weather API**: Open-Meteo (free)
- **Data Processing**: Pandas, NumPy

## Model Details

**LSTM Architecture:**
- Input: 30-day lookback window
- 2 LSTM layers (64 → 32 units)
- Dropout: 0.2
- Output: 7-day forecast

**Features:**
- Historical sales (revenue, order count)
- Day of week (one-hot encoded)
- Weekend/holiday flags
- Temperature and precipitation
- Campaign activity

## Business Value

1. **Reduce Waste**: Accurate forecasts minimize over-ordering
2. **Prevent Stockouts**: Smart buffers ensure availability
3. **Optimize Labor**: Plan prep schedules efficiently
4. **Data-Driven Decisions**: Replace gut instinct with ML insights

## Team

Built for the Deloitte x AUC Hackathon 2024
