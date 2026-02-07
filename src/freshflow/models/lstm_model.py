"""
LSTM model architecture for demand forecasting.
Implements a 2-layer LSTM with dropout for time series prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import os


class DemandLSTM(nn.Module):
    """
    LSTM neural network for demand forecasting.

    Architecture:
    - Input: (batch_size, sequence_length, n_features)
    - LSTM Layer 1: 64 units with dropout
    - LSTM Layer 2: 32 units with dropout
    - Dense Layer: 16 units with ReLU
    - Output: (batch_size, forecast_horizon)
    """

    def __init__(
        self,
        n_features: int = 12,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dense_size: int = 16,
        forecast_horizon: int = 7,
        dropout: float = 0.2
    ):
        super(DemandLSTM, self).__init__()

        self.n_features = n_features
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.forecast_horizon = forecast_horizon

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size_1,
            batch_first=True,
            dropout=dropout if hidden_size_2 > 0 else 0
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)

        Returns:
            Predictions of shape (batch_size, forecast_horizon)
        """
        # LSTM layers
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout(lstm_out1)

        lstm_out2, _ = self.lstm2(lstm_out1)

        # Take last timestep output
        last_output = lstm_out2[:, -1, :]

        # Dense layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class DemandPredictor:
    """
    Wrapper class for making predictions with trained LSTM model.
    Handles data preprocessing, model loading, and inference.
    """

    def __init__(
        self,
        model: Optional[DemandLSTM] = None,
        scaler_params: Optional[dict] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model
        self.scaler_params = scaler_params or {}

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def predict(
        self,
        sequence: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions from input sequence.

        Args:
            sequence: Input sequence of shape (sequence_length, n_features)
            return_confidence: Whether to return confidence intervals

        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Prepare input
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(x).cpu().numpy()[0]

        # Denormalize if scaler params available
        if "mean" in self.scaler_params and "std" in self.scaler_params:
            predictions = predictions * self.scaler_params["std"] + self.scaler_params["mean"]

        # Calculate confidence intervals (simplified - using fixed percentage)
        if return_confidence:
            # Use 10% of prediction as confidence interval
            confidence = np.abs(predictions) * 0.1
            return predictions, confidence

        return predictions, None

    def save(self, filepath: str) -> None:
        """Save model and scaler parameters."""
        if self.model is None:
            raise ValueError("No model to save")

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "n_features": self.model.n_features,
                "hidden_size_1": self.model.hidden_size_1,
                "hidden_size_2": self.model.hidden_size_2,
                "forecast_horizon": self.model.forecast_horizon
            },
            "scaler_params": self.scaler_params
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(save_dict, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "DemandPredictor":
        """Load model from file."""
        save_dict = torch.load(filepath, map_location=device)

        config = save_dict["model_config"]
        model = DemandLSTM(
            n_features=config["n_features"],
            hidden_size_1=config["hidden_size_1"],
            hidden_size_2=config["hidden_size_2"],
            forecast_horizon=config["forecast_horizon"]
        )
        model.load_state_dict(save_dict["model_state_dict"])

        return cls(
            model=model,
            scaler_params=save_dict.get("scaler_params", {}),
            device=device
        )


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 30,
    forecast_horizon: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and targets for training.

    Args:
        data: Input data of shape (n_samples, n_features)
               First column should be the target variable
        sequence_length: Number of past days to use
        forecast_horizon: Number of days to predict

    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []

    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i + sequence_length])
        # Target is the first column (target variable) for next forecast_horizon days
        y.append(data[i + sequence_length:i + sequence_length + forecast_horizon, 0])

    return np.array(X), np.array(y)


def normalize_data(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data using z-score normalization.

    Args:
        data: Input data
        mean: Pre-computed mean (for inference)
        std: Pre-computed std (for inference)

    Returns:
        Tuple of (normalized_data, mean, std)
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero

    normalized = (data - mean) / std
    return normalized, mean, std
