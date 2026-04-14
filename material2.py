"""
material2.py - Model Classes and Trainers for Stock Prediction
Contains: LSTM, GRU, RNN models, ARIMA trainer, and Deep Learning trainer
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import random
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
# Set device and seed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 62

def set_seed(seed=62):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==================== Deep Learning Models ====================

class LSTMModel(nn.Module):
    """LSTM Model with optional bidirectional and layer normalization"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3,
                 bidirectional=False, output_dim=1, use_layernorm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_layernorm = use_layernorm

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        if self.use_layernorm:
            last_out = self.layer_norm(last_out)
        out = self.fc(last_out)
        return out


class GRUModel(nn.Module):
    """GRU Model for sequence prediction"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class RNNModel(nn.Module):
    """Basic RNN Model"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.0, nonlinearity='tanh'):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers>1 else 0.0,
            nonlinearity=nonlinearity
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)
#===================================== Transformer Model ====================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, hidden_dim))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        out = self.transformer_encoder(x)
        out = self.fc(out[:, -1, :])
        return out
# ==================== Dataset ====================

class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ==================== Early Stopping ====================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.stop = False
    
    def step(self, loss):
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True






class MLTrainer:
    """Trainer for traditional ML models (Leakage-Free, Stock Safe)"""

    def __init__(self, model_type, seq_len=60, **model_params):
        self.model_type = model_type
        self.seq_len = seq_len
        self.model_params = model_params

        # Separate scalers
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Initialize model
        if model_type == "Random Forest":
            self.model = RandomForestRegressor(**model_params)
        elif model_type == "XGBoost":
            self.model = XGBRegressor(**model_params)
        elif model_type == "LightGBM":
            self.model = LGBMRegressor(**model_params, verbose=-1)
      #  elif model_type == "CatBoost":
        #    self.model = CatBoostRegressor(**model_params, verbose=0)
        elif model_type == "Gradient Boosting":
            self.model = GradientBoostingRegressor(**model_params)
        elif model_type == "Linear Regression":
            self.model = LinearRegression(**model_params)
        elif model_type == "Ridge":
            self.model = Ridge(**model_params)
        elif model_type == "Lasso":
            self.model = Lasso(**model_params)
        elif model_type == "ElasticNet":
            self.model = ElasticNet(**model_params)
        elif model_type == "SVR":
            self.model = SVR(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # -----------------------------
    # Prepare data WITHOUT LEAKAGE - FIXED VERSION
    # -----------------------------
    def prepare_data(self, data):
        """
        Prepare data with NO LEAKAGE:
        1. Split with GAP
        2. Fit scalers on TRAIN only
        3. Create sequences AFTER scaling
        """
        data = np.asarray(data)
        
        print(f"\n{'='*60}")
        print(f"DATA PREPARATION DEBUG - FIXED VERSION")
        print(f"{'='*60}")
        print(f"Input data shape: {data.shape}")
        print(f"Input data range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"First 3 rows:\n{data[:3]}")
        
        # Step 1: Split with GAP to prevent leakage
        split_idx = int(len(data) * 0.8)
        train_raw = data[:split_idx - self.seq_len]
        val_raw = data[split_idx - self.seq_len:]
        
        print(f"\nLEAKAGE-FREE SPLIT:")
        print(f"   Original 80% split: {split_idx}")
        print(f"   Train ends at: {split_idx - self.seq_len}")
        print(f"   Gap: {self.seq_len} rows")
        print(f"   Train size: {len(train_raw)} | Val size: {len(val_raw)}")
        
        # Step 2: Fit scaler on TRAIN data only
        train_scaled = self.scaler_X.fit_transform(train_raw)
        
        # Step 3: Transform validation data
        val_scaled = self.scaler_X.transform(val_raw)
        
        print(f"\nAfter scaling:")
        print(f"   Train range: [{train_scaled.min():.4f}, {train_scaled.max():.4f}]")
        print(f"   Val range: [{val_scaled.min():.4f}, {val_scaled.max():.4f}]")
        
        # Step 4: Create sequences (val_raw already includes seq_len context)
        X_train, y_train = self._create_sequences(train_scaled)
        X_val, y_val = self._create_sequences(val_scaled)
        
        print(f"\nSequences created:")
        print(f"   X_train: {X_train.shape} | y_train: {y_train.shape}")
        print(f"   X_val: {X_val.shape} | y_val: {y_val.shape}")
        
        # Step 5: Scale targets (fit on train only)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        
        print(f"\nTargets after scaling:")
        print(f"   y_train range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
        print(f"   y_val range: [{y_val_scaled.min():.4f}, {y_val_scaled.max():.4f}]")
        
        print(f"{'='*60}\n")
        
        return X_train, y_train_scaled, X_val, y_val_scaled

    def _create_sequences(self, data):
        """Create sequences from scaled data"""
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            # Input: data from i to i+seq_len
            X.append(data[i:i + self.seq_len].flatten())
            # Target: NEXT day's Close price (i + seq_len)
            y.append(data[i + self.seq_len, 0])
        return np.array(X), np.array(y)

    # -----------------------------
    # Train model
    # -----------------------------
    def train(self, data, progress_bar=None, status_text=None):
        """Train the model with progress tracking"""
        if status_text:
            status_text.text(f"Preparing data for {self.model_type}...")
        
        X_train, y_train, X_val, y_val = self.prepare_data(data)
        
        if progress_bar:
            progress_bar.progress(0.3)
        if status_text:
            status_text.text(f"Training {self.model_type} model...")

        # Train the model
        self.model.fit(X_train, y_train)

        if progress_bar:
            progress_bar.progress(0.7)

        # Validate
        y_pred_scaled = self.model.predict(X_val)

        # Inverse transform to get actual prices
        y_true = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("Training completed")

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        return metrics

    # -----------------------------
    # Predict future WITHOUT LEAKAGE
    # -----------------------------
    def predict(self, last_sequence, n_steps):
        """
        Predict future prices WITHOUT LEAKAGE:
        1. Use only the last seq_len rows from last_sequence
        2. Scale using the fitted scaler
        3. For multi-step: use recursive prediction with realistic feature propagation
        """
        last_sequence = np.asarray(last_sequence)
        
        # Take last seq_len rows and scale them
        current_seq = last_sequence[-self.seq_len:]
        current_seq_scaled = self.scaler_X.transform(current_seq)
        
        predictions_scaled = []
        
        for step in range(n_steps):
            # Flatten and predict
            x_input = current_seq_scaled.flatten().reshape(1, -1)
            pred_scaled = self.model.predict(x_input)[0]
            predictions_scaled.append(pred_scaled)
            
            # Create next input sequence
            # CRITICAL: We need to update the FEATURE space (scaler_X), not target space (scaler_y)
            # Strategy: Copy last row's features, then update Close price position
            
            # Get the last unscaled row to maintain feature relationships
            last_unscaled = current_seq[-1].copy()
            
            # Update Close price with our prediction (inverse transform from target space)
            predicted_price_unscaled = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            last_unscaled[0] = predicted_price_unscaled
            
            # Now scale this new row using feature scaler
            new_row_scaled = self.scaler_X.transform([last_unscaled])[0]
            
            # Roll the window: drop first row, add new row
            current_seq_scaled = np.vstack([current_seq_scaled[1:], new_row_scaled])
            current_seq = np.vstack([current_seq[1:], last_unscaled])
        
        # Inverse transform predictions to get actual prices
        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions_scaled).ravel()
        
        return predictions

    # -----------------------------
    # Metrics
    # -----------------------------
    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics with safety"""
        # Ensure positive values for MAPE
        mask = y_true > 0.01  # Only use prices > $0.01
        
        if mask.sum() == 0:  # All prices too small
            mape = np.nan
        else:
            y_true_safe = y_true[mask]
            y_pred_safe = y_pred[mask]
            mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100
        
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": mape if not np.isnan(mape) else 0.0
        }
# ==================== ARIMA Trainer ====================

class ARIMATrainer:
    """Trainer for ARIMA family models (AR, MA, ARMA, ARIMA)"""
    def __init__(self, order=(1, 0, 0), model_type='ARIMA'):
        """
        Args:
            order: (p, d, q) for ARIMA
            model_type: 'AR', 'MA', 'ARMA', 'ARIMA'
        """
        self.order = order
        self.model_type = model_type
        self.model = None
        self.model_fit = None
        self.train_data = None
        
    def train(self, data, progress_bar, status_text):
        """Train ARIMA model"""
        prices = data[:, 0]  # Use first column (Close price)
        
        # Split train/val
        split_idx = int(len(prices) * 0.8)
        self.train_data = prices[:split_idx]
        val_data = prices[split_idx:]
        
        status_text.text(f"Training {self.model_type} model...")
        progress_bar.progress(0.3)
        
        try:
            # Fit model
            if self.model_type == 'AR':
                self.model = AutoReg(self.train_data, lags=self.order[0])
                self.model_fit = self.model.fit()
            else:
                self.model = ARIMA(self.train_data, order=self.order)
                self.model_fit = self.model.fit()
            
            progress_bar.progress(0.6)
            
            # Predict validation set (vectorized - much faster!)
            n_forecast = len(val_data)
            
            if self.model_type == 'AR':
                start_idx = len(self.train_data)
                end_idx = len(self.train_data) + n_forecast - 1
                val_predictions = self.model_fit.predict(start=start_idx, end=end_idx)
            else:
                val_predictions = self.model_fit.forecast(steps=n_forecast)
            
            progress_bar.progress(1.0)
            status_text.text("Training completed")
            
            val_predictions = np.array(val_predictions)
            
            # Calculate metrics
            mse = mean_squared_error(val_data, val_predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(val_data, val_predictions)
            r2 = r2_score(val_data, val_predictions)
            
            residuals = val_data - val_predictions
            mape = np.mean(np.abs(residuals / np.clip(val_data, 1e-8, None))) * 100
            
            return {
                "rmse": rmse,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "mape": mape,
                "y_true": val_data.reshape(-1, 1),
                "y_pred": val_predictions.reshape(-1, 1),
                "train_data": self.train_data,
                "val_data": val_data
            }
            
        except Exception as e:
            status_text.text(f"Error: {str(e)}")
            raise e
    
    def predict(self, history, n_steps=1):
        """Predict future values - optimized version"""
        all_data = history[:, 0]  # Use Close price
        
        # Refit model on all available data (one time only)
        if self.model_type == 'AR':
            model_temp = AutoReg(all_data, lags=self.order[0])
            model_fit_temp = model_temp.fit()
            # Predict all steps at once
            start_idx = len(all_data)
            end_idx = len(all_data) + n_steps - 1
            predictions = model_fit_temp.predict(start=start_idx, end=end_idx)
        else:
            model_temp = ARIMA(all_data, order=self.order)
            model_fit_temp = model_temp.fit()
            # Forecast all steps at once (much faster!)
            predictions = model_fit_temp.forecast(steps=n_steps)
        
        return np.array(predictions)


# ==================== Deep Learning Trainer ====================

class TimeSeriesTrainer:
    """Trainer for deep learning models (LSTM, GRU, RNN)"""
    def __init__(self, model, seq_len=20, batch_size=32, epochs=50, lr=0.001,
                 criterion=None, optimizer=None, early_stopping=True, patience=20):
        self.model = model
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.criterion = criterion if criterion else nn.MSELoss()
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=lr)

        self.train_losses = []
        self.val_losses = []
        self.early_stopping = EarlyStopping(patience=patience) if early_stopping else None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def create_sequences(self, X, y):
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_len):
            X_seq.append(X[i:i+self.seq_len])
            y_seq.append(y[i+self.seq_len])
        return np.array(X_seq), np.array(y_seq)

    def train(self, data, progress_bar, status_text):
        """Train the model"""
        X = data
        y = data[:, 0].reshape(-1, 1)

        # Split train/val
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]

        # Scale data
        X_train = self.scaler_X.fit_transform(X_train_raw)
        X_val = self.scaler_X.transform(X_val_raw)
        y_train = self.scaler_y.fit_transform(y_train_raw)
        y_val = self.scaler_y.transform(y_val_raw)

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)

        # Create data loaders
        train_loader = DataLoader(
            StockDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                        torch.tensor(y_train_seq, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            StockDataset(torch.tensor(X_val_seq, dtype=torch.float32),
                        torch.tensor(y_val_seq, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False
        )

        self.model.to(DEVICE)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_epoch_loss = []

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())

            avg_train_loss = np.mean(train_epoch_loss)
            self.train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            val_epoch_loss = []
            val_preds_scaled = []
            val_true_scaled = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = self.model(xb)
                    val_epoch_loss.append(self.criterion(pred, yb).item())
                    val_preds_scaled.extend(pred.cpu().numpy())
                    val_true_scaled.extend(yb.cpu().numpy())

            avg_val_loss = np.mean(val_epoch_loss)
            self.val_losses.append(avg_val_loss)

            # Early stopping
            if self.early_stopping:
                self.early_stopping.step(avg_val_loss)
                if self.early_stopping.stop:
                    status_text.text(f"⏹ Early stopping at epoch {epoch+1}")
                    break

            # Update progress
            progress = (epoch + 1) / self.epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Calculate final metrics
        val_preds_real = self.scaler_y.inverse_transform(np.array(val_preds_scaled))
        val_true_real = self.scaler_y.inverse_transform(np.array(val_true_scaled))

        mse = mean_squared_error(val_true_real, val_preds_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(val_true_real, val_preds_real)
        r2 = r2_score(val_true_real, val_preds_real)
        
        residuals = val_true_real - val_preds_real
        mape = np.mean(np.abs(residuals / np.clip(val_true_real, 1e-8, None))) * 100

        return {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "mape": mape,
            "y_true": val_true_real,
            "y_pred": val_preds_real,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }

    def predict(self, last_seq, n_steps=1):
        """Predict future values"""
        self.model.eval()
        seq = self.scaler_X.transform(last_seq)
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        preds = []

        with torch.no_grad():
            for _ in range(n_steps):
                pred = self.model(seq)
                preds.append(pred.cpu().numpy().ravel()[0])

                # Update sequence
                last_features = seq[:, -1, :].cpu().numpy()
                next_step = last_features.copy()
                next_step[0, 0] = pred.cpu().numpy().ravel()[0]
                seq = torch.cat([seq[:, 1:, :], torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).to(DEVICE)], dim=1)

        preds_real = self.scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
        return preds_real
def plot_forecast(y_true, y_pred, dates=None, model_name="Model",
                  zoom_last_n=120, figsize=(16, 10), save_path=None,
                  show_confidence=True, confidence_level=0.95):
    # Data preparation
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    if dates is None:
        dates = np.arange(n)
    else:
        dates = np.asarray(dates)[-n:]

    # Calculate metrics
    residuals = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(residuals / np.clip(y_true, 1e-8, None))) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    max_error = np.max(np.abs(residuals))
    median_ae = np.median(np.abs(residuals))
    
    # Directional accuracy (for time series)
    if n > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = None

    # Confidence intervals
    if show_confidence:
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_upper = y_pred + z_score * std_residual
        ci_lower = y_pred - z_score * std_residual

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # 1. Full forecast with confidence interval
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, y_true, label='Actual', linewidth=2, color='black', alpha=0.8)
    ax1.plot(dates, y_pred, label='Predicted', linewidth=1.5, linestyle='--', color='red', alpha=0.7)
    
    if show_confidence:
        ax1.fill_between(dates, ci_lower, ci_upper, alpha=0.2, color='red', 
                         label=f'{int(confidence_level*100)}% Confidence Interval')
    else:
        ax1.fill_between(dates, y_true, y_pred, alpha=0.2, color='red')
    
    ax1.set_title(f'{model_name} - Full Forecast | MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%',
                  fontweight='bold', fontsize=12)
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Rotate date labels if dates are provided
    if not isinstance(dates[0], (int, np.integer)):
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Zoomed view
    ax2 = fig.add_subplot(gs[1, :])
    zoom_n = min(zoom_last_n, n)
    zoom_dates = dates[-zoom_n:]
    
    ax2.plot(zoom_dates, y_true[-zoom_n:], label='Actual',
             linewidth=2, color='black', marker='o', markersize=3, alpha=0.8)
    ax2.plot(zoom_dates, y_pred[-zoom_n:], label='Predicted',
             linewidth=1.5, linestyle='--', color='red', marker='s', markersize=3, alpha=0.7)
    
    if show_confidence:
        ax2.fill_between(zoom_dates, ci_lower[-zoom_n:], ci_upper[-zoom_n:], 
                         alpha=0.2, color='red')
    
    ax2.set_title(f'Zoomed View - Last {zoom_n} Points', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Price', fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    if not isinstance(zoom_dates[0], (int, np.integer)):
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Residuals over time
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(range(n), residuals, alpha=0.6, color='steelblue', width=1.0)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.axhline(mean_residual, color='orange', linestyle=':', linewidth=1.5, 
                label=f'Mean: {mean_residual:.4f}')
    ax3.set_title('Residuals Over Time', fontweight='bold', fontsize=10)
    ax3.set_xlabel('Sample Index', fontsize=9)
    ax3.set_ylabel('Residual', fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Actual vs Predicted scatter
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction')
    ax4.set_xlabel('Actual', fontsize=9)
    ax4.set_ylabel('Predicted', fontsize=9)
    ax4.set_title(f'Actual vs Predicted (R²={r2:.4f})', fontweight='bold', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Residual distribution
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    
    # Overlay normal distribution
    mu, sigma = mean_residual, std_residual
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax5.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Dist')
    
    ax5.axvline(0, color='green', linestyle='--', linewidth=1.5, label='Zero Error')
    ax5.set_title('Residual Distribution', fontweight='bold', fontsize=10)
    ax5.set_xlabel('Residual', fontsize=9)
    ax5.set_ylabel('Density', fontsize=9)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()

    # Print comprehensive summary
    print(f"\n{'='*70}")
    print(f"  {model_name} - Comprehensive Performance Summary")
    print(f"{'='*70}")
    print(f"  Error Metrics:")
    print(f"     MAE:           {mae:.4f}")
    print(f"     RMSE:          {rmse:.4f}")
    print(f"     MAPE:          {mape:.2f}%")
    print(f"     Max Error:     {max_error:.4f}")
    print(f"     Median AE:     {median_ae:.4f}")
    print(f"\n  Goodness of Fit:")
    print(f"     R² Score:      {r2:.4f}")
    if directional_accuracy is not None:
        print(f"     Dir. Accuracy: {directional_accuracy:.2f}%")
    print(f"\n  Residual Statistics:")
    print(f"     Mean:          {mean_residual:.4f}")
    print(f"     Std Dev:       {std_residual:.4f}")
    print(f"     Skewness:      {stats.skew(residuals):.4f}")
    print(f"     Kurtosis:      {stats.kurtosis(residuals):.4f}")
    print(f"\n   Data Info:")
    print(f"     Sample Size:   {n}")
    print(f"     Actual Range:  [{y_true.min():.4f}, {y_true.max():.4f}]")
    print(f"     Pred Range:    [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"{'='*70}\n")
    
    # Return metrics dictionary
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'max_error': max_error,
        'median_ae': median_ae,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'directional_accuracy': directional_accuracy,
        'n_samples': n
    }
    
    return fig, metrics


# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    
    # Create synthetic price data
    y_true = 100 + np.cumsum(np.random.randn(n_samples) * 2)
    y_pred = y_true + np.random.randn(n_samples) * 5
    
    # Create date range
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    
    # Plot with all features
    fig, metrics = plot_forecast(
        y_true, 
        y_pred, 
        dates=dates,
        model_name="LSTM",
        zoom_last_n=50,
        figsize=(16, 10),
        show_confidence=True,
        confidence_level=0.95
    )
    
    # Access metrics
    print(f"\nReturned metrics: {metrics}")