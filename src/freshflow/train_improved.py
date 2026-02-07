"""
FreshFlow - Train Improved Ensemble Model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from freshflow.services.data_processor import DataProcessor
from freshflow.services.weather_service import WeatherService
from freshflow.models.improved_model import EnsembleForecaster
from freshflow.utils.helpers import MODELS_DIR, ensure_dir


def train_ensemble(place_id: int, epochs: int = 100, verbose: bool = True):
    """Train improved ensemble model for a location."""
    print("=" * 60)
    print(f"🌿 FreshFlow - Improved Ensemble Training")
    print("=" * 60)

    # Load data
    print("\n📊 Loading data...")
    dp = DataProcessor()
    ws = WeatherService()

    daily_sales = dp.get_daily_sales(place_id)
    print(f"   Location ID: {place_id}")
    print(f"   Days of data: {len(daily_sales)}")

    # Get location name
    places = dp.load_places()
    place_name = places[places['id'] == place_id]['title'].values
    name = place_name[0] if len(place_name) > 0 else 'Unknown'
    print(f"   Location: {name}")

    # Get weather
    min_date = daily_sales['date'].min()
    max_date = daily_sales['date'].max()
    weather_df = ws.get_historical_weather(
        min_date.to_pydatetime(),
        max_date.to_pydatetime()
    )

    # Create and train model
    print("\n🚀 Training ensemble model...")
    print("   Components: Attention-LSTM + Gradient Boosting")
    print("-" * 60)

    model = EnsembleForecaster(
        n_features=34,  # Will be updated during training
        sequence_length=30,
        forecast_horizon=7,
        lstm_weight=0.6
    )

    metrics, history = model.train(
        daily_sales=daily_sales,
        weather_df=weather_df,
        epochs=epochs,
        batch_size=16,
        lr=0.001,
        verbose=verbose
    )

    # Print results
    print("-" * 60)
    print("\n✅ Training complete!")
    print("\n📊 Model Comparison:")
    print(f"   {'Model':<20} {'MAPE':>10} {'RMSE':>12} {'R²':>10}")
    print(f"   {'-'*52}")
    print(f"   {'Attention-LSTM':<20} {metrics['lstm_mape']:>9.1%} {metrics['lstm_rmse']:>11,.0f} {metrics['lstm_r2']:>10.3f}")
    print(f"   {'Gradient Boosting':<20} {metrics['gb_mape']:>9.1%} {metrics['gb_rmse']:>11,.0f} {metrics['gb_r2']:>10.3f}")
    print(f"   {'Ensemble (60/40)':<20} {metrics['ensemble_mape']:>9.1%} {metrics['ensemble_rmse']:>11,.0f} {metrics['ensemble_r2']:>10.3f}")

    # Save model
    ensure_dir(MODELS_DIR)
    save_path = os.path.join(MODELS_DIR, f"ensemble_model_{place_id}.pt")
    model.save(save_path)
    print(f"\n💾 Model saved to: {save_path}")

    # Save metrics
    import json
    metrics_path = os.path.join(MODELS_DIR, f"ensemble_model_{place_id}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    return model, metrics


def train_top_locations(n_locations: int = 3, epochs: int = 80):
    """Train ensemble models for top locations."""
    dp = DataProcessor()
    locations = dp.get_locations().head(n_locations)

    all_results = {}

    for _, loc in locations.iterrows():
        place_id = loc['place_id']

        try:
            model, metrics = train_ensemble(place_id, epochs=epochs, verbose=False)
            all_results[place_id] = {
                'name': loc['place_name'],
                'orders': loc['order_count'],
                'ensemble_mape': metrics['ensemble_mape'],
                'ensemble_r2': metrics['ensemble_r2']
            }
        except Exception as e:
            print(f"❌ Failed for {loc['place_name']}: {e}")
            all_results[place_id] = {'error': str(e)}

        print("\n")

    # Summary
    print("=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"\n{'Location':<35} {'Orders':>10} {'MAPE':>10} {'R²':>10}")
    print("-" * 65)

    for place_id, result in all_results.items():
        if 'error' not in result:
            print(f"{result['name'][:34]:<35} {result['orders']:>10,} {result['ensemble_mape']:>9.1%} {result['ensemble_r2']:>10.3f}")
        else:
            print(f"{place_id:<35} {'ERROR':>10} {'-':>10} {'-':>10}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--place-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--top", type=int, default=0, help="Train top N locations")

    args = parser.parse_args()

    if args.top > 0:
        train_top_locations(args.top, args.epochs)
    elif args.place_id:
        train_ensemble(args.place_id, args.epochs)
    else:
        # Default: train on Restaurant China (best previous results)
        train_ensemble(552477, args.epochs)
