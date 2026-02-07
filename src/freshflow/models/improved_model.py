"""
FreshFlow - Improved Demand Forecasting Model
Features:
- Enhanced feature engineering with lagged features
- LSTM with attention mechanism
- Ensemble with gradient boosting
- Better regularization and training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class Attention(nn.Module):
    """Attention mechanism for LSTM outputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism and residual connections."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        forecast_horizon: int = 7
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention
        self.attention = Attention(hidden_size * 2)

        # Output layers with residual
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        context, _ = self.attention(lstm_out)

        # Output
        return self.fc(context)


class FeatureEngineerV2:
    """Enhanced feature engineering with lagged features and rolling statistics."""

    def __init__(self):
        self.scaler = RobustScaler()
        self.target_scaler = RobustScaler()

    def create_features(
        self,
        daily_sales: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Create enhanced feature set."""
        df = daily_sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Target: log transform
        df['log_revenue'] = np.log1p(df['total_revenue'])
        df['log_orders'] = np.log1p(df['order_count'])

        # Lagged features (1, 7, 14, 28 days)
        for lag in [1, 7, 14, 28]:
            df[f'revenue_lag_{lag}'] = df['log_revenue'].shift(lag)
            df[f'orders_lag_{lag}'] = df['log_orders'].shift(lag)

        # Rolling statistics (7, 14, 28 day windows)
        for window in [7, 14, 28]:
            df[f'revenue_roll_mean_{window}'] = df['log_revenue'].rolling(window).mean()
            df[f'revenue_roll_std_{window}'] = df['log_revenue'].rolling(window).std()
            df[f'orders_roll_mean_{window}'] = df['log_orders'].rolling(window).mean()

        # Trend features
        df['revenue_diff_1'] = df['log_revenue'].diff(1)
        df['revenue_diff_7'] = df['log_revenue'].diff(7)

        # Day of week (cyclical encoding)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month (cyclical encoding)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Week of year
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

        # Binary features
        df['is_weekend'] = df['is_weekend'].astype(float)
        df['is_holiday'] = df['is_holiday'].astype(float)

        # Day of month (for payday effects)
        day_of_month = df['date'].dt.day
        df['is_month_start'] = (day_of_month <= 5).astype(float)
        df['is_month_end'] = (day_of_month >= 25).astype(float)

        # Weather features if available
        if weather_df is not None:
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            df = df.merge(
                weather_df[['date', 'temp_mean', 'precipitation']],
                on='date',
                how='left'
            )
            df['temp_mean'] = df['temp_mean'].fillna(df['temp_mean'].median())
            df['precipitation'] = df['precipitation'].fillna(0)

            # Weather interactions
            df['temp_weekend'] = df['temp_mean'] * df['is_weekend']
            df['rain_flag'] = (df['precipitation'] > 1).astype(float)
        else:
            df['temp_mean'] = 10.0
            df['precipitation'] = 0.0
            df['temp_weekend'] = 0.0
            df['rain_flag'] = 0.0

        # Drop rows with NaN from lagging
        df = df.dropna().reset_index(drop=True)

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return [
            'log_revenue', 'log_orders',
            'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14', 'revenue_lag_28',
            'orders_lag_1', 'orders_lag_7', 'orders_lag_14', 'orders_lag_28',
            'revenue_roll_mean_7', 'revenue_roll_std_7',
            'revenue_roll_mean_14', 'revenue_roll_std_14',
            'revenue_roll_mean_28',
            'orders_roll_mean_7', 'orders_roll_mean_14', 'orders_roll_mean_28',
            'revenue_diff_1', 'revenue_diff_7',
            'dow_sin', 'dow_cos',
            'month_sin', 'month_cos',
            'week_sin', 'week_cos',
            'is_weekend', 'is_holiday',
            'is_month_start', 'is_month_end',
            'temp_mean', 'precipitation', 'temp_weekend', 'rain_flag'
        ]

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        forecast_horizon: int = 7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for training."""
        feature_cols = self.get_feature_columns()
        features = df[feature_cols].values

        # Fit scalers
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)

        # Target is log_revenue (first column)
        target = df['log_revenue'].values
        self.target_scaler.fit(target.reshape(-1, 1))

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length - forecast_horizon + 1):
            X.append(scaled_features[i:i + sequence_length])
            y.append(target[i + sequence_length:i + sequence_length + forecast_horizon])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Scale targets
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

        return X, y, y_scaled, features


class EnsembleForecaster:
    """Ensemble model combining LSTM with Gradient Boosting."""

    def __init__(
        self,
        n_features: int,
        sequence_length: int = 30,
        forecast_horizon: int = 7,
        lstm_weight: float = 0.6
    ):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_weight = lstm_weight

        self.lstm_model = AttentionLSTM(
            n_features=n_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            forecast_horizon=forecast_horizon
        )

        # One GB model per forecast day
        self.gb_models = [
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            for _ in range(forecast_horizon)
        ]

        self.feature_engineer = FeatureEngineerV2()
        self.is_trained = False

    def train(
        self,
        daily_sales: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.001,
        verbose: bool = True
    ) -> Dict:
        """Train the ensemble model."""
        print("📊 Preparing enhanced features...")
        df = self.feature_engineer.create_features(daily_sales, weather_df)
        print(f"   Samples after feature engineering: {len(df)}")

        X, y_raw, y_scaled, flat_features = self.feature_engineer.prepare_sequences(
            df, self.sequence_length, self.forecast_horizon
        )

        print(f"   Sequences created: {len(X)}")
        print(f"   Features per timestep: {X.shape[2]}")

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train_raw, y_val_raw = y_raw[:split], y_raw[split:]
        y_train_scaled, y_val_scaled = y_scaled[:split], y_scaled[split:]

        # ===== Train LSTM =====
        print("\n🧠 Training Attention-LSTM...")
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train_scaled)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val_scaled)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.AdamW(
            self.lstm_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float('inf')
        patience = 0
        max_patience = 20
        history = []

        for epoch in range(epochs):
            # Train
            self.lstm_model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self.lstm_model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            scheduler.step()

            # Validate
            self.lstm_model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    pred = self.lstm_model(X_batch)
                    val_loss += criterion(pred, y_batch).item()
            val_loss /= len(val_loader)

            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

            if verbose and (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.lstm_model.state_dict().copy()
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"   Early stopping at epoch {epoch + 1}")
                    break

        self.lstm_model.load_state_dict(best_state)

        # ===== Train Gradient Boosting =====
        print("\n🌲 Training Gradient Boosting ensemble...")

        # Flatten sequences for GB (use last timestep features + aggregates)
        X_gb_train = self._prepare_gb_features(X_train)
        X_gb_val = self._prepare_gb_features(X_val)

        for day in range(self.forecast_horizon):
            self.gb_models[day].fit(X_gb_train, y_train_raw[:, day])

        # ===== Evaluate Ensemble =====
        print("\n📈 Evaluating ensemble...")

        # Get predictions
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred_scaled = self.lstm_model(torch.FloatTensor(X_val)).numpy()

        # Inverse transform LSTM predictions
        lstm_pred = self.feature_engineer.target_scaler.inverse_transform(
            lstm_pred_scaled.reshape(-1, 1)
        ).reshape(lstm_pred_scaled.shape)

        # GB predictions
        gb_pred = np.column_stack([
            self.gb_models[day].predict(X_gb_val)
            for day in range(self.forecast_horizon)
        ])

        # Ensemble
        ensemble_pred = self.lstm_weight * lstm_pred + (1 - self.lstm_weight) * gb_pred

        # Convert from log scale
        y_val_exp = np.expm1(y_val_raw)
        lstm_pred_exp = np.expm1(np.clip(lstm_pred, 0, 20))
        gb_pred_exp = np.expm1(np.clip(gb_pred, 0, 20))
        ensemble_pred_exp = np.expm1(np.clip(ensemble_pred, 0, 20))

        # Calculate metrics
        def calc_mape(actual, pred):
            mask = actual > 0
            return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask]))

        def calc_rmse(actual, pred):
            return np.sqrt(np.mean((actual - pred) ** 2))

        def calc_r2(actual, pred):
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics = {
            'lstm_mape': calc_mape(y_val_exp.flatten(), lstm_pred_exp.flatten()),
            'lstm_rmse': calc_rmse(y_val_exp.flatten(), lstm_pred_exp.flatten()),
            'lstm_r2': calc_r2(y_val_exp.flatten(), lstm_pred_exp.flatten()),
            'gb_mape': calc_mape(y_val_exp.flatten(), gb_pred_exp.flatten()),
            'gb_rmse': calc_rmse(y_val_exp.flatten(), gb_pred_exp.flatten()),
            'gb_r2': calc_r2(y_val_exp.flatten(), gb_pred_exp.flatten()),
            'ensemble_mape': calc_mape(y_val_exp.flatten(), ensemble_pred_exp.flatten()),
            'ensemble_rmse': calc_rmse(y_val_exp.flatten(), ensemble_pred_exp.flatten()),
            'ensemble_r2': calc_r2(y_val_exp.flatten(), ensemble_pred_exp.flatten()),
        }

        self.is_trained = True

        return metrics, history

    def _prepare_gb_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare features for gradient boosting (flatten + aggregate)."""
        # Use last timestep
        last_step = X[:, -1, :]

        # Add aggregates
        mean_features = X.mean(axis=1)
        std_features = X.std(axis=1)
        min_features = X.min(axis=1)
        max_features = X.max(axis=1)

        return np.hstack([last_step, mean_features, std_features, min_features, max_features])

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions from ensemble."""
        self.lstm_model.eval()

        with torch.no_grad():
            lstm_pred_scaled = self.lstm_model(torch.FloatTensor(X)).numpy()

        lstm_pred = self.feature_engineer.target_scaler.inverse_transform(
            lstm_pred_scaled.reshape(-1, 1)
        ).reshape(lstm_pred_scaled.shape)

        X_gb = self._prepare_gb_features(X)
        gb_pred = np.column_stack([
            self.gb_models[day].predict(X_gb)
            for day in range(self.forecast_horizon)
        ])

        ensemble_pred = self.lstm_weight * lstm_pred + (1 - self.lstm_weight) * gb_pred

        # Convert from log
        return np.expm1(lstm_pred), np.expm1(gb_pred), np.expm1(ensemble_pred)

    def save(self, path: str) -> None:
        """Save the ensemble model."""
        import pickle
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            'lstm_state': self.lstm_model.state_dict(),
            'gb_models': self.gb_models,
            'feature_engineer': self.feature_engineer,
            'n_features': self.n_features,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'lstm_weight': self.lstm_weight
        }

        torch.save(save_dict, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> 'EnsembleForecaster':
        """Load a saved ensemble model."""
        save_dict = torch.load(path, map_location='cpu', weights_only=False)

        model = cls(
            n_features=save_dict['n_features'],
            sequence_length=save_dict['sequence_length'],
            forecast_horizon=save_dict['forecast_horizon'],
            lstm_weight=save_dict['lstm_weight']
        )

        model.lstm_model.load_state_dict(save_dict['lstm_state'])
        model.gb_models = save_dict['gb_models']
        model.feature_engineer = save_dict['feature_engineer']
        model.is_trained = True

        return model
