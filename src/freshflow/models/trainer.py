"""
Training pipeline for demand forecasting LSTM model.
Handles data preparation, training loop, and model evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

from .lstm_model import DemandLSTM, DemandPredictor, create_sequences, normalize_data
from ..utils.helpers import calculate_mape, calculate_rmse, calculate_r2, MODELS_DIR, ensure_dir


class DemandTrainer:
    """
    Trains LSTM model for demand forecasting.
    """

    def __init__(
        self,
        n_features: int = 12,
        sequence_length: int = 30,
        forecast_horizon: int = 7,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = "cpu"
    ):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device)

        self.model: Optional[DemandLSTM] = None
        self.scaler_params: Dict = {}
        self.training_history: List[Dict] = []

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Prepare data for training.

        Args:
            features_df: Feature matrix from FeatureEngineer
            train_ratio: Ratio of data for training

        Returns:
            Tuple of (train_loader, val_loader, scaler_params)
        """
        # Convert to numpy
        data = features_df.values.astype(np.float32)

        # Normalize
        normalized_data, mean, std = normalize_data(data)

        # Store scaler params (for target column)
        self.scaler_params = {
            "mean": mean[0],
            "std": std[0],
            "full_mean": mean,
            "full_std": std
        }

        # Create sequences
        X, y = create_sequences(
            normalized_data,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon
        )

        # Update n_features based on actual data
        self.n_features = X.shape[2]

        # Train/validation split
        split_idx = int(len(X) * train_ratio)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, self.scaler_params

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict:
        """
        Train the LSTM model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Print training progress

        Returns:
            Dictionary with training results
        """
        # Initialize model
        self.model = DemandLSTM(
            n_features=self.n_features,
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            forecast_horizon=self.forecast_horizon
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        self.training_history = []

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()

                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(y_batch.cpu().numpy())

            val_loss /= len(val_loader)

            # Calculate metrics (denormalized)
            val_pred_denorm = np.array(val_predictions) * self.scaler_params["std"] + self.scaler_params["mean"]
            val_target_denorm = np.array(val_targets) * self.scaler_params["std"] + self.scaler_params["mean"]

            # Flatten for metric calculation
            val_pred_flat = val_pred_denorm.flatten()
            val_target_flat = val_target_denorm.flatten()

            mape = calculate_mape(val_target_flat, val_pred_flat)
            rmse = calculate_rmse(val_target_flat, val_pred_flat)
            r2 = calculate_r2(val_target_flat, val_pred_flat)

            # Log history
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mape": mape,
                "rmse": rmse,
                "r2": r2
            })

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"MAPE: {mape:.2%}, R2: {r2:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if "best_model_state" in dir():
            self.model.load_state_dict(best_model_state)

        # Final metrics
        final_metrics = self.training_history[-1] if self.training_history else {}

        return {
            "final_train_loss": final_metrics.get("train_loss", 0),
            "final_val_loss": final_metrics.get("val_loss", 0),
            "final_mape": final_metrics.get("mape", 0),
            "final_rmse": final_metrics.get("rmse", 0),
            "final_r2": final_metrics.get("r2", 0),
            "epochs_trained": len(self.training_history),
            "history": self.training_history
        }

    def get_predictor(self) -> DemandPredictor:
        """Get a predictor instance with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return DemandPredictor(
            model=self.model,
            scaler_params=self.scaler_params,
            device=str(self.device)
        )

    def save_model(self, name: str = "demand_model") -> str:
        """
        Save trained model to file.

        Args:
            name: Model name (without extension)

        Returns:
            Path to saved model
        """
        ensure_dir(MODELS_DIR)
        filepath = os.path.join(MODELS_DIR, f"{name}.pt")

        predictor = self.get_predictor()
        predictor.save(filepath)

        return filepath

    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                predictions.extend(pred.cpu().numpy())
                targets.extend(y_batch.numpy())

        # Denormalize
        predictions = np.array(predictions) * self.scaler_params["std"] + self.scaler_params["mean"]
        targets = np.array(targets) * self.scaler_params["std"] + self.scaler_params["mean"]

        # Flatten
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        return {
            "mape": calculate_mape(target_flat, pred_flat),
            "rmse": calculate_rmse(target_flat, pred_flat),
            "r2": calculate_r2(target_flat, pred_flat),
            "predictions": predictions,
            "targets": targets
        }


def quick_train(
    features_df: pd.DataFrame,
    epochs: int = 50,
    verbose: bool = True
) -> Tuple[DemandPredictor, Dict]:
    """
    Quick training function for simple use cases.

    Args:
        features_df: Feature matrix from FeatureEngineer
        epochs: Number of training epochs
        verbose: Print progress

    Returns:
        Tuple of (predictor, training_results)
    """
    trainer = DemandTrainer(
        n_features=features_df.shape[1],
        epochs=epochs
    )

    train_loader, val_loader, _ = trainer.prepare_data(features_df)
    results = trainer.train(train_loader, val_loader, verbose=verbose)

    predictor = trainer.get_predictor()

    return predictor, results
