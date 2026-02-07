"""
FreshFlow - LSTM Model Training Script
Trains a demand forecasting model on historical sales data.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from freshflow.services.data_processor import DataProcessor
from freshflow.services.weather_service import WeatherService
from freshflow.services.feature_engineer import FeatureEngineer
from freshflow.models.trainer import DemandTrainer
from freshflow.utils.helpers import MODELS_DIR, ensure_dir


def train_demand_model(
    place_id: int = None,
    epochs: int = 50,
    batch_size: int = 32,
    sequence_length: int = 30,
    forecast_horizon: int = 7,
    verbose: bool = True
):
    """
    Train LSTM model for demand forecasting.

    Args:
        place_id: Specific location ID or None for aggregate model
        epochs: Number of training epochs
        batch_size: Training batch size
        sequence_length: Days of history to use for prediction
        forecast_horizon: Days ahead to forecast
        verbose: Print training progress
    """
    print("=" * 60)
    print("🌿 FreshFlow - LSTM Model Training")
    print("=" * 60)

    # Initialize services
    print("\n📊 Loading data...")
    data_processor = DataProcessor()
    weather_service = WeatherService()
    feature_engineer = FeatureEngineer(data_processor, weather_service)

    # Load and display data stats
    orders = data_processor.load_orders()
    print(f"   Total orders: {len(orders):,}")
    print(f"   Date range: {orders['date'].min()} to {orders['date'].max()}")
    print(f"   Unique locations: {orders['place_id'].nunique()}")

    # Create features
    print("\n🔧 Engineering features...")
    features_df, feature_names = feature_engineer.create_training_features(
        place_id=place_id,
        target_column="total_revenue"
    )
    print(f"   Feature matrix shape: {features_df.shape}")
    print(f"   Features: {feature_names}")

    # Initialize trainer
    print("\n🧠 Initializing LSTM model...")
    trainer = DemandTrainer(
        n_features=len(feature_names),
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        hidden_size_1=64,
        hidden_size_2=32,
        learning_rate=0.001,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=10,
        device="cpu"  # Use "cuda" if GPU available
    )

    # Prepare data
    print("\n📦 Preparing training data...")
    train_loader, val_loader, scaler_params = trainer.prepare_data(
        features_df,
        train_ratio=0.8
    )
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    # Train model
    print("\n🚀 Training model...")
    print("-" * 60)

    results = trainer.train(train_loader, val_loader, verbose=verbose)

    print("-" * 60)
    print("\n✅ Training complete!")
    print(f"   Epochs trained: {results['epochs_trained']}")
    print(f"   Final MAPE: {results['final_mape']:.2%}")
    print(f"   Final RMSE: {results['final_rmse']:,.2f} DKK")
    print(f"   Final R²: {results['final_r2']:.4f}")

    # Save model
    print("\n💾 Saving model...")
    ensure_dir(MODELS_DIR)

    model_name = f"demand_model_{'all' if place_id is None else place_id}"
    model_path = trainer.save_model(model_name)
    print(f"   Model saved to: {model_path}")

    # Save training history
    history_df = pd.DataFrame(results['history'])
    history_path = os.path.join(MODELS_DIR, f"{model_name}_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"   History saved to: {history_path}")

    print("\n" + "=" * 60)
    print("🎉 Model ready for use!")
    print("=" * 60)

    return trainer, results


def train_top_locations(n_locations: int = 5, epochs: int = 30):
    """Train individual models for top N locations."""
    print("\n🏪 Training models for top locations...")

    data_processor = DataProcessor()
    locations = data_processor.get_locations().head(n_locations)

    results = {}

    for _, loc in locations.iterrows():
        place_id = loc['place_id']
        place_name = loc['place_name']

        print(f"\n\n{'='*60}")
        print(f"📍 Training model for: {place_name[:40]}")
        print(f"   Place ID: {place_id}")
        print(f"   Orders: {loc['order_count']:,}")

        try:
            trainer, result = train_demand_model(
                place_id=place_id,
                epochs=epochs,
                verbose=False
            )
            results[place_id] = {
                'name': place_name,
                'mape': result['final_mape'],
                'r2': result['final_r2']
            }
            print(f"   ✅ MAPE: {result['final_mape']:.2%}, R²: {result['final_r2']:.4f}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[place_id] = {'name': place_name, 'error': str(e)}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train FreshFlow demand forecasting model")
    parser.add_argument("--place-id", type=int, default=None,
                       help="Specific place ID (None for aggregate)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--sequence-length", type=int, default=30,
                       help="Days of history for prediction")
    parser.add_argument("--forecast-horizon", type=int, default=7,
                       help="Days ahead to forecast")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--top-locations", type=int, default=0,
                       help="Train models for top N locations")

    args = parser.parse_args()

    if args.top_locations > 0:
        train_top_locations(args.top_locations, args.epochs)
    else:
        train_demand_model(
            place_id=args.place_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            forecast_horizon=args.forecast_horizon,
            verbose=not args.quiet
        )
