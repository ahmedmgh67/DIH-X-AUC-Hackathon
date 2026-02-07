"""
FreshFlow - Model Training Dashboard
Train new models, view performance, and manage model versions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import glob
import pickle
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from freshflow.services.data_processor import DataProcessor
from freshflow.utils.helpers import format_currency

st.set_page_config(page_title="Model Training - FreshFlow", page_icon="🧠", layout="wide")


@st.cache_resource
def get_services():
    dp = DataProcessor()
    return dp


def get_model_info():
    """Get information about all trained models."""
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models"
    )

    models = []

    if not os.path.exists(models_dir):
        return models

    # Find all GB models
    for model_file in glob.glob(os.path.join(models_dir, "gb_model_*.pkl")):
        try:
            filename = os.path.basename(model_file)
            place_id = filename.replace("gb_model_", "").replace(".pkl", "")
            place_id = int(float(place_id))

            # Load metrics if available
            metrics_file = os.path.join(models_dir, f"gb_metrics_{place_id}.json")
            metrics = {}
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

            # Get file modification time
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))

            # Calculate accuracy, clamping to 0-100 range
            mape = metrics.get('mape', None)
            if mape is not None:
                accuracy = max(0, min(100, 100 - mape))
            else:
                accuracy = None

            models.append({
                'place_id': place_id,
                'model_type': 'Gradient Boosting',
                'filename': filename,
                'trained_at': mod_time,
                'mape': mape,
                'rmse': metrics.get('rmse', None),
                'r2': metrics.get('r2', None),
                'accuracy': accuracy,
                'samples': metrics.get('train_samples', None)
            })
        except Exception as e:
            continue

    # Find LSTM models
    for model_file in glob.glob(os.path.join(models_dir, "*_v2.pt")):
        try:
            filename = os.path.basename(model_file)
            place_id = filename.replace("_v2.pt", "")
            place_id = int(float(place_id))

            # Check if already have GB model for this location
            if any(m['place_id'] == place_id and m['model_type'] == 'Gradient Boosting' for m in models):
                continue

            mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))

            models.append({
                'place_id': place_id,
                'model_type': 'LSTM',
                'filename': filename,
                'trained_at': mod_time,
                'mape': None,
                'rmse': None,
                'r2': None,
                'accuracy': None,
                'samples': None
            })
        except Exception as e:
            continue

    return models


def get_trainable_locations(dp):
    """Get locations that can be trained."""
    locations = dp.get_locations()
    orders = dp.load_orders()

    # Count days and revenue per location from orders
    orders['date'] = pd.to_datetime(orders['created'], unit='s').dt.date
    location_stats = orders.groupby('place_id').agg({
        'date': 'nunique',  # Count unique days
        'total_amount': ['mean', 'std']
    }).reset_index()
    location_stats.columns = ['place_id', 'days', 'avg_revenue', 'std_revenue']

    # Merge with location names
    result = locations.merge(location_stats, on='place_id', how='inner')
    result = result.sort_values('days', ascending=False)

    return result


def train_model_for_location(place_id: int, dp) -> dict:
    """Train a Gradient Boosting model for a specific location."""
    from freshflow.models.gradient_boost_model import OptimizedGBForecaster

    # Initialize forecaster
    forecaster = OptimizedGBForecaster()

    # Get training data
    daily_sales = dp.get_daily_sales(place_id)

    if len(daily_sales) < 60:
        return {'success': False, 'error': 'Insufficient data (need at least 60 days)'}

    # Train the model
    metrics = forecaster.train(daily_sales)

    # Save the model
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models"
    )
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"gb_model_{place_id}.pkl")
    metrics_path = os.path.join(models_dir, f"gb_metrics_{place_id}.json")

    forecaster.save(model_path)

    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    return {
        'success': True,
        'metrics': metrics,
        'model_path': model_path
    }


def main():
    dp = get_services()

    st.markdown("# 🧠 Model Training Dashboard")
    st.markdown("Train, manage, and monitor demand forecasting models")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("### Quick Stats")
        models = get_model_info()

        st.metric("Trained Models", len(models))

        if models:
            production_ready = [m for m in models if m['accuracy'] and m['accuracy'] >= 70]
            st.metric("Production Ready", len(production_ready),
                     help="Models with 70%+ accuracy")

            if production_ready:
                avg_accuracy = sum(m['accuracy'] for m in production_ready) / len(production_ready)
                st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")

        st.divider()
        st.page_link("app.py", label="← Back to Dashboard", icon="🏠")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Overview", "🚀 Train New Model", "📈 Performance Analysis", "⚙️ Model Management"
    ])

    with tab1:
        st.subheader("📊 Trained Models Overview")

        models = get_model_info()

        if models:
            # Summary cards
            col1, col2, col3, col4 = st.columns(4)

            gb_models = [m for m in models if m['model_type'] == 'Gradient Boosting']
            lstm_models = [m for m in models if m['model_type'] == 'LSTM']

            with col1:
                st.metric("Total Models", len(models))
            with col2:
                st.metric("GB Models", len(gb_models))
            with col3:
                st.metric("LSTM Models", len(lstm_models))
            with col4:
                high_accuracy = [m for m in models if m['accuracy'] and m['accuracy'] >= 90]
                st.metric("High Accuracy (90%+)", len(high_accuracy))

            # Models table
            st.markdown("### Model List")

            # Get location names
            locations = dp.get_locations()

            model_df = pd.DataFrame(models)
            model_df = model_df.merge(
                locations[['place_id', 'place_name']],
                on='place_id',
                how='left'
            )

            # Format for display
            display_df = model_df[[
                'place_name', 'model_type', 'accuracy', 'mape', 'r2', 'trained_at', 'samples'
            ]].copy()

            display_df['accuracy'] = display_df['accuracy'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
            )
            display_df['mape'] = display_df['mape'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )
            display_df['r2'] = display_df['r2'].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
            display_df['trained_at'] = display_df['trained_at'].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else "N/A"
            )
            display_df['samples'] = display_df['samples'].apply(
                lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
            )

            display_df.columns = ['Location', 'Model Type', 'Accuracy', 'MAPE', 'R²', 'Trained At', 'Samples']

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Accuracy distribution chart
            st.markdown("### Accuracy Distribution")

            models_with_accuracy = [m for m in models if m['accuracy'] is not None]
            if models_with_accuracy:
                acc_df = pd.DataFrame(models_with_accuracy)
                acc_df = acc_df.merge(locations[['place_id', 'place_name']], on='place_id', how='left')

                fig = px.bar(
                    acc_df.sort_values('accuracy', ascending=True),
                    x='accuracy',
                    y='place_name',
                    orientation='h',
                    color='accuracy',
                    color_continuous_scale='RdYlGn',
                    range_color=[50, 100]
                )
                fig.add_vline(x=70, line_dash="dash", line_color="orange",
                             annotation_text="Production threshold (70%)")
                fig.update_layout(
                    xaxis_title="Accuracy (%)",
                    yaxis_title="Location",
                    height=max(400, len(models_with_accuracy) * 30)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trained models found. Go to 'Train New Model' to train your first model!")

    with tab2:
        st.subheader("🚀 Train New Model")

        # Get trainable locations
        trainable = get_trainable_locations(dp)

        # Filter to locations with enough data
        trainable = trainable[trainable['days'] >= 60]

        if len(trainable) == 0:
            st.warning("No locations with sufficient data (60+ days) for training.")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### Available Locations")
                st.caption("Locations with 60+ days of data can be trained")

                # Show locations with training status
                models = get_model_info()
                trained_place_ids = [m['place_id'] for m in models]

                trainable['status'] = trainable['place_id'].apply(
                    lambda x: '✅ Trained' if x in trained_place_ids else '⏳ Not trained'
                )
                trainable['avg_revenue'] = trainable['avg_revenue'].apply(
                    lambda x: f"{x:,.0f} DKK"
                )

                display_trainable = trainable[[
                    'place_name', 'days', 'avg_revenue', 'status'
                ]].copy()
                display_trainable.columns = ['Location', 'Days of Data', 'Avg Daily Revenue', 'Status']

                st.dataframe(display_trainable, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("### Train a Model")

                # Location selector
                location_options = trainable['place_name'].tolist()
                selected_location = st.selectbox("Select Location", location_options)

                if selected_location:
                    place_id = trainable[trainable['place_name'] == selected_location]['place_id'].values[0]
                    days = trainable[trainable['place_name'] == selected_location]['days'].values[0]

                    st.info(f"📊 {int(days)} days of training data available")

                    # Check if already trained
                    if place_id in trained_place_ids:
                        st.warning("⚠️ This location already has a trained model. Training will replace it.")

                    if st.button("🚀 Start Training", type="primary"):
                        with st.spinner(f"Training model for {selected_location}..."):
                            result = train_model_for_location(place_id, dp)

                            if result['success']:
                                st.success("✅ Model trained successfully!")

                                metrics = result['metrics']

                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    accuracy = 100 - metrics.get('mape', 0)
                                    st.metric("Accuracy", f"{accuracy:.1f}%")
                                with col_b:
                                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                                with col_c:
                                    st.metric("R²", f"{metrics.get('r2', 0):.3f}")

                                if accuracy >= 70:
                                    st.success("🎉 Model is production-ready!")
                                else:
                                    st.warning("⚠️ Model accuracy is below 70%. Consider gathering more data.")

                                st.cache_resource.clear()
                            else:
                                st.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

    with tab3:
        st.subheader("📈 Performance Analysis")

        models = get_model_info()
        models_with_metrics = [m for m in models if m['mape'] is not None]

        if models_with_metrics:
            locations = dp.get_locations()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### MAPE by Location")

                mape_df = pd.DataFrame(models_with_metrics)
                mape_df = mape_df.merge(locations[['place_id', 'place_name']], on='place_id', how='left')

                fig = px.bar(
                    mape_df.sort_values('mape'),
                    x='place_name',
                    y='mape',
                    color='mape',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(
                    xaxis_title="Location",
                    yaxis_title="MAPE (%)",
                    xaxis_tickangle=-45,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### R² Score by Location")

                r2_df = mape_df[mape_df['r2'].notna()]

                if len(r2_df) > 0:
                    fig = px.bar(
                        r2_df.sort_values('r2', ascending=False),
                        x='place_name',
                        y='r2',
                        color='r2',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        xaxis_title="Location",
                        yaxis_title="R² Score",
                        xaxis_tickangle=-45,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Model comparison
            st.markdown("### Model Quality Tiers")

            tier_counts = {
                'Excellent (90%+)': len([m for m in models_with_metrics if m['accuracy'] >= 90]),
                'Good (80-90%)': len([m for m in models_with_metrics if 80 <= m['accuracy'] < 90]),
                'Fair (70-80%)': len([m for m in models_with_metrics if 70 <= m['accuracy'] < 80]),
                'Needs Improvement (<70%)': len([m for m in models_with_metrics if m['accuracy'] < 70])
            }

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏆 Excellent", tier_counts['Excellent (90%+)'])
            with col2:
                st.metric("👍 Good", tier_counts['Good (80-90%)'])
            with col3:
                st.metric("📊 Fair", tier_counts['Fair (70-80%)'])
            with col4:
                st.metric("⚠️ Needs Work", tier_counts['Needs Improvement (<70%)'])

            # Pie chart
            fig = px.pie(
                values=list(tier_counts.values()),
                names=list(tier_counts.keys()),
                color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No models with performance metrics available. Train models to see performance analysis.")

    with tab4:
        st.subheader("⚙️ Model Management")

        models = get_model_info()

        if models:
            locations = dp.get_locations()

            # Model selector
            model_df = pd.DataFrame(models)
            model_df = model_df.merge(locations[['place_id', 'place_name']], on='place_id', how='left')

            selected_model = st.selectbox(
                "Select Model",
                model_df['place_name'].tolist(),
                format_func=lambda x: f"{x} ({model_df[model_df['place_name'] == x]['model_type'].values[0]})"
            )

            if selected_model:
                model_info = model_df[model_df['place_name'] == selected_model].iloc[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Model Details")
                    st.markdown(f"**Location:** {model_info['place_name']}")
                    st.markdown(f"**Model Type:** {model_info['model_type']}")
                    st.markdown(f"**Trained:** {model_info['trained_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"**File:** `{model_info['filename']}`")

                    if model_info['accuracy']:
                        st.markdown(f"**Accuracy:** {model_info['accuracy']:.1f}%")
                    if model_info['mape']:
                        st.markdown(f"**MAPE:** {model_info['mape']:.2f}%")
                    if model_info['r2']:
                        st.markdown(f"**R²:** {model_info['r2']:.3f}")

                with col2:
                    st.markdown("### Actions")

                    place_id = model_info['place_id']

                    if st.button("🔄 Retrain Model", type="primary"):
                        with st.spinner(f"Retraining model for {selected_model}..."):
                            result = train_model_for_location(place_id, dp)

                            if result['success']:
                                st.success("✅ Model retrained successfully!")
                                metrics = result['metrics']
                                st.metric("New Accuracy", f"{100 - metrics.get('mape', 0):.1f}%")
                                st.cache_resource.clear()
                            else:
                                st.error(f"❌ Retraining failed: {result.get('error', 'Unknown error')}")

                    st.divider()

                    st.markdown("### Danger Zone")
                    if st.button("🗑️ Delete Model", type="secondary"):
                        models_dir = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "saved_models"
                        )
                        model_path = os.path.join(models_dir, model_info['filename'])
                        metrics_path = os.path.join(models_dir, f"gb_metrics_{place_id}.json")

                        try:
                            if os.path.exists(model_path):
                                os.remove(model_path)
                            if os.path.exists(metrics_path):
                                os.remove(metrics_path)
                            st.success(f"✅ Model deleted: {model_info['filename']}")
                            st.cache_resource.clear()
                        except Exception as e:
                            st.error(f"❌ Error deleting model: {str(e)}")

            # Batch operations
            st.divider()
            st.markdown("### Batch Operations")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Retrain All Models"):
                    progress = st.progress(0)
                    status = st.empty()

                    for i, model in enumerate(models):
                        status.text(f"Retraining {model['place_id']}...")
                        train_model_for_location(model['place_id'], dp)
                        progress.progress((i + 1) / len(models))

                    status.text("✅ All models retrained!")
                    st.cache_resource.clear()

            with col2:
                # Export model info
                export_data = model_df[[
                    'place_name', 'model_type', 'accuracy', 'mape', 'r2', 'trained_at'
                ]].copy()

                csv = export_data.to_csv(index=False)
                st.download_button(
                    "📥 Export Model Report",
                    csv,
                    f"model_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
        else:
            st.info("No models available. Train some models first!")


if __name__ == "__main__":
    main()
