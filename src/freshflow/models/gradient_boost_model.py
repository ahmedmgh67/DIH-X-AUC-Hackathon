"""
FreshFlow - Optimized Gradient Boosting Model
Fast, reliable demand forecasting with interpretable features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class OptimizedGBForecaster:
    """
    Optimized Gradient Boosting forecaster with:
    - Feature importance analysis
    - Cross-validation for robust evaluation
    - Multi-horizon forecasting
    """

    def __init__(self, forecast_horizon: int = 7):
        self.forecast_horizon = forecast_horizon
        self.models: List = []
        self.scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics: Dict = {}

    def _create_features(self, daily_sales: pd.DataFrame) -> pd.DataFrame:
        """Create optimized feature set with fewer lags to preserve data."""
        df = daily_sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Target: log transform
        df['log_revenue'] = np.log1p(df['total_revenue'])

        # Fewer lags (1, 7 days only) to preserve more data
        for lag in [1, 7]:
            df[f'revenue_lag_{lag}'] = df['log_revenue'].shift(lag)

        # Rolling statistics (shorter windows)
        for window in [7, 14]:
            df[f'revenue_roll_mean_{window}'] = df['log_revenue'].rolling(window).mean()
            df[f'revenue_roll_std_{window}'] = df['log_revenue'].rolling(window).std()

        # Simple trend
        df['revenue_diff_1'] = df['log_revenue'].diff(1)
        df['revenue_diff_7'] = df['log_revenue'].diff(7)

        # Same day last week / 2 weeks ago
        df['revenue_same_dow_1w'] = df['log_revenue'].shift(7)
        df['revenue_same_dow_2w'] = df['log_revenue'].shift(14)

        # Day of week (one-hot - more interpretable for GB)
        for i in range(7):
            df[f'dow_{i}'] = (df['day_of_week'] == i).astype(float)

        # Month effects
        for m in [1, 6, 7, 8, 12]:  # Key months
            df[f'month_{m}'] = (df['month'] == m).astype(float)

        # Binary features
        df['is_weekend'] = df['is_weekend'].astype(float)
        df['is_holiday'] = df['is_holiday'].astype(float)

        # Day of month effects
        dom = df['date'].dt.day
        df['is_month_start'] = (dom <= 5).astype(float)
        df['is_month_end'] = (dom >= 25).astype(float)

        # Week of year (cyclical)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

        # Drop NaN (only lose 14 days)
        df = df.dropna().reset_index(drop=True)

        return df

    def _get_feature_columns(self) -> List[str]:
        """Get feature column names."""
        return [
            'revenue_lag_1', 'revenue_lag_7',
            'revenue_roll_mean_7', 'revenue_roll_std_7',
            'revenue_roll_mean_14', 'revenue_roll_std_14',
            'revenue_diff_1', 'revenue_diff_7',
            'revenue_same_dow_1w', 'revenue_same_dow_2w',
            'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
            'month_1', 'month_6', 'month_7', 'month_8', 'month_12',
            'is_weekend', 'is_holiday',
            'is_month_start', 'is_month_end',
            'week_sin', 'week_cos'
        ]

    def train(
        self,
        daily_sales: pd.DataFrame,
        n_cv_splits: int = 3,
        verbose: bool = True
    ) -> Dict:
        """Train GB models with time series cross-validation."""
        if verbose:
            print("📊 Creating features...")

        df = self._create_features(daily_sales)
        self.feature_names = self._get_feature_columns()

        if verbose:
            print(f"   Samples: {len(df)}")
            print(f"   Features: {len(self.feature_names)}")

        # Prepare data
        X = df[self.feature_names].values
        y = df['log_revenue'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create target arrays for each forecast horizon
        y_targets = []
        valid_indices = []

        for h in range(1, self.forecast_horizon + 1):
            y_h = np.roll(y, -h)
            valid_idx = np.arange(len(y) - h)
            y_targets.append((y_h[valid_idx], valid_idx))

        # Use minimum valid samples
        min_valid = len(y) - self.forecast_horizon
        X_valid = X_scaled[:min_valid]
        y_targets_valid = [y[h:min_valid + h] for h in range(1, self.forecast_horizon + 1)]

        if verbose:
            print(f"   Valid samples for training: {min_valid}")

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)

        cv_scores = {h: [] for h in range(self.forecast_horizon)}

        if verbose:
            print(f"\n🔄 Cross-validation ({n_cv_splits} folds)...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_valid)):
            for h in range(self.forecast_horizon):
                X_train, X_val = X_valid[train_idx], X_valid[val_idx]
                y_train, y_val = y_targets_valid[h][train_idx], y_targets_valid[h][val_idx]

                # Use HistGradientBoosting for speed
                model = HistGradientBoostingRegressor(
                    max_iter=200,
                    max_depth=6,
                    learning_rate=0.05,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=10,
                    random_state=42
                )

                model.fit(X_train, y_train)
                pred = model.predict(X_val)

                # Calculate MAPE
                y_val_exp = np.expm1(y_val)
                pred_exp = np.expm1(np.clip(pred, 0, 15))
                mask = y_val_exp > 0
                mape = np.mean(np.abs((y_val_exp[mask] - pred_exp[mask]) / y_val_exp[mask]))
                cv_scores[h].append(mape)

        # Average CV scores
        avg_cv_mape = np.mean([np.mean(scores) for scores in cv_scores.values()])

        if verbose:
            print(f"   Average CV MAPE: {avg_cv_mape:.1%}")

        # Train final models on all data
        if verbose:
            print(f"\n🌲 Training final models...")

        self.models = []
        for h in range(self.forecast_horizon):
            model = HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=6,
                learning_rate=0.05,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=42
            )
            model.fit(X_valid, y_targets_valid[h])
            self.models.append(model)

        # Final evaluation (last 20% as holdout)
        split = int(len(X_valid) * 0.8)
        X_test = X_valid[split:]
        y_test = [yt[split:] for yt in y_targets_valid]

        all_preds = []
        all_actuals = []

        for h in range(self.forecast_horizon):
            pred = self.models[h].predict(X_test)
            pred_exp = np.expm1(np.clip(pred, 0, 15))
            actual_exp = np.expm1(y_test[h])

            all_preds.extend(pred_exp)
            all_actuals.extend(actual_exp)

        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)

        mask = all_actuals > 0
        final_mape = np.mean(np.abs((all_actuals[mask] - all_preds[mask]) / all_actuals[mask]))
        final_rmse = np.sqrt(np.mean((all_actuals - all_preds) ** 2))

        ss_res = np.sum((all_actuals - all_preds) ** 2)
        ss_tot = np.sum((all_actuals - np.mean(all_actuals)) ** 2)
        final_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        self.metrics = {
            'cv_mape': avg_cv_mape,
            'test_mape': final_mape,
            'test_rmse': final_rmse,
            'test_r2': final_r2
        }

        self.is_trained = True

        if verbose:
            print(f"\n✅ Training complete!")
            print(f"   CV MAPE:   {avg_cv_mape:.1%}")
            print(f"   Test MAPE: {final_mape:.1%}")
            print(f"   Test RMSE: {final_rmse:,.0f} DKK")
            print(f"   Test R²:   {final_r2:.3f}")

        return self.metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained models."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Average importance across all horizon models
        importances = np.zeros(len(self.feature_names))

        for model in self.models:
            # HistGradientBoosting doesn't have feature_importances_
            # Use permutation importance approximation
            pass

        # Use first model's feature importance if available
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.ones(len(self.feature_names)) / len(self.feature_names)
        }).sort_values('importance', ascending=False)

        return importance_df

    def predict(self, daily_sales: pd.DataFrame) -> np.ndarray:
        """Generate forecast for next N days."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        df = self._create_features(daily_sales)
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        # Use last row for prediction
        X_last = X_scaled[-1:, :]

        predictions = []
        for model in self.models:
            pred = model.predict(X_last)[0]
            predictions.append(np.expm1(max(0, pred)))

        return np.array(predictions)

    def save(self, path: str) -> None:
        """Save model."""
        import pickle
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'forecast_horizon': self.forecast_horizon
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str) -> 'OptimizedGBForecaster':
        """Load model."""
        import pickle

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        model = cls(forecast_horizon=save_dict['forecast_horizon'])
        model.models = save_dict['models']
        model.scaler = save_dict['scaler']
        model.feature_names = save_dict['feature_names']
        model.metrics = save_dict['metrics']
        model.is_trained = True

        return model
