"""
FreshFlow - Improved LSTM Model Training
Uses log-transform and better normalization for revenue forecasting.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from freshflow.services.data_processor import DataProcessor
from freshflow.services.weather_service import WeatherService
from freshflow.utils.helpers import MODELS_DIR, ensure_dir, calculate_mape, calculate_rmse, calculate_r2


class ImprovedLSTM(nn.Module):
    """Improved LSTM with residual connections."""

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3, forecast_horizon=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


def prepare_data(place_id, sequence_length=30, forecast_horizon=7):
    """Prepare data with improved preprocessing."""
    print("📊 Loading data...")
    dp = DataProcessor()
    ws = WeatherService()

    # Get daily sales for location
    daily = dp.get_daily_sales(place_id)
    print(f"   Days of data: {len(daily)}")

    # Get weather data
    min_date = daily['date'].min()
    max_date = daily['date'].max()
    weather = ws.get_historical_weather(min_date.to_pydatetime(), max_date.to_pydatetime())

    # Merge
    daily['date'] = pd.to_datetime(daily['date'])
    weather['date'] = pd.to_datetime(weather['date'])
    merged = daily.merge(weather[['date', 'temp_mean', 'precipitation']], on='date', how='left')

    # Fill missing weather
    merged['temp_mean'] = merged['temp_mean'].fillna(merged['temp_mean'].median())
    merged['precipitation'] = merged['precipitation'].fillna(0)

    # Log transform revenue (add 1 to handle zeros)
    merged['log_revenue'] = np.log1p(merged['total_revenue'])
    merged['log_orders'] = np.log1p(merged['order_count'])

    # Create feature matrix
    features = merged[[
        'log_revenue', 'log_orders',
        'day_of_week', 'is_weekend', 'is_holiday',
        'temp_mean', 'precipitation'
    ]].copy()

    # One-hot encode day of week
    for i in range(7):
        features[f'dow_{i}'] = (features['day_of_week'] == i).astype(float)
    features = features.drop('day_of_week', axis=1)

    # Add month cyclical
    month = merged['month'].values
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Normalize features
    data = features.values.astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1
    normalized = (data - mean) / std

    # Create sequences
    X, y = [], []
    for i in range(len(normalized) - sequence_length - forecast_horizon + 1):
        X.append(normalized[i:i + sequence_length])
        # Target: log revenue for next forecast_horizon days
        y.append(normalized[i + sequence_length:i + sequence_length + forecast_horizon, 0])

    X = np.array(X)
    y = np.array(y)

    print(f"   Sequences created: {len(X)}")
    print(f"   Feature count: {X.shape[2]}")

    return X, y, mean, std, merged


def train_improved_model(place_id, epochs=100, batch_size=16, lr=0.001):
    """Train improved model."""
    print("=" * 60)
    print("🌿 FreshFlow - Improved LSTM Training")
    print("=" * 60)

    # Prepare data
    X, y, mean, std, daily_data = prepare_data(place_id)

    # Split data
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Create dataloaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"\n📦 Data split:")
    print(f"   Train: {len(X_train)} sequences")
    print(f"   Val: {len(X_val)} sequences")

    # Create model
    n_features = X.shape[2]
    model = ImprovedLSTM(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print(f"\n🚀 Training for {epochs} epochs...")
    print("-" * 60)

    best_val_loss = float('inf')
    patience = 0
    max_patience = 15
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                val_preds.extend(pred.numpy())
                val_targets.extend(y_batch.numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Denormalize for metrics
        val_preds = np.array(val_preds) * std[0] + mean[0]
        val_targets = np.array(val_targets) * std[0] + mean[0]

        # Convert from log
        val_preds_exp = np.expm1(val_preds).flatten()
        val_targets_exp = np.expm1(val_targets).flatten()

        # Clip negatives
        val_preds_exp = np.clip(val_preds_exp, 0, None)

        mape = calculate_mape(val_targets_exp, val_preds_exp)
        rmse = calculate_rmse(val_targets_exp, val_preds_exp)
        r2 = calculate_r2(val_targets_exp, val_preds_exp)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mape': mape,
            'rmse': rmse,
            'r2': r2
        })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAPE: {mape:.1%} | R²: {r2:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            best_metrics = {'mape': mape, 'rmse': rmse, 'r2': r2}
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_state)

    print("-" * 60)
    print(f"\n✅ Training complete!")
    print(f"   Best MAPE: {best_metrics['mape']:.1%}")
    print(f"   Best RMSE: {best_metrics['rmse']:,.0f} DKK")
    print(f"   Best R²: {best_metrics['r2']:.3f}")

    # Save model
    ensure_dir(MODELS_DIR)
    save_path = os.path.join(MODELS_DIR, f"demand_model_{place_id}_v2.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std,
        'n_features': n_features,
        'place_id': place_id,
        'metrics': best_metrics
    }, save_path)

    print(f"\n💾 Model saved to: {save_path}")

    # Save history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(MODELS_DIR, f"demand_model_{place_id}_v2_history.csv")
    history_df.to_csv(history_path, index=False)

    return model, best_metrics, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--place-id", type=int, default=94025,
                       help="Location ID (default: Kaffestuen Vesterbro)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    train_improved_model(
        place_id=args.place_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
