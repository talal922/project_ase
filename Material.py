# ==================== TimeSeries Flexible Trainer ====================
import torch 
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import random
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 62
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3,
                 bidirectional=False, output_dim=1, use_layernorm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_layernorm = use_layernorm

        # LSTM 
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional ######### BI
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

# ----------------- GRU Model -----------------
class GRUModel(nn.Module):
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
# ----------------- RNN Model -----------------

class RNNModel(nn.Module):
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
class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ---------- EarlyStopping ----------
class EarlyStopping:
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

# ---------- Trainer ----------
class TimeSeriesTrainerSimple:
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
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_len):
            X_seq.append(X[i:i+self.seq_len])
            y_seq.append(y[i+self.seq_len])
        return np.array(X_seq), np.array(y_seq)

    def direction_accuracy(self, y_true, y_pred):
        # Up/Down accuracy
        return np.mean(np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])) * 100

    def train(self, data):
        # ---------------- Prepare data ----------------
        X = data
        y = data[:, 0].reshape(-1, 1)  

        # Split train/val BEFORE scaling to avoid leakage
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]

        # @@@@@@@@@@@@@@@@@@
        X_train = self.scaler_X.fit_transform(X_train_raw)
        X_val = self.scaler_X.transform(X_val_raw)
        y_train = self.scaler_y.fit_transform(y_train_raw)
        y_val = self.scaler_y.transform(y_val_raw)

      

       
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)

        train_loader = DataLoader(StockDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                               torch.tensor(y_train_seq, dtype=torch.float32)),
                                  batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(StockDataset(torch.tensor(X_val_seq, dtype=torch.float32),
                                             torch.tensor(y_val_seq, dtype=torch.float32)),
                                batch_size=self.batch_size, shuffle=False)

        self.model.to(DEVICE)

        for epoch in range(self.epochs):
            self.model.train()
            train_epoch_loss = []
            train_preds_scaled = []
            train_true_scaled = []

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())

                train_preds_scaled.extend(pred.detach().cpu().numpy())
                train_true_scaled.extend(yb.detach().cpu().numpy())

            avg_train_loss = np.mean(train_epoch_loss)
            self.train_losses.append(avg_train_loss)

            # ---------- Validation ----------
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
                    print(f"\n⏹ Early stopping at epoch {epoch+1}")
                    break

            # ---------- Epoch logging ----------
            train_preds_real = self.scaler_y.inverse_transform(np.array(train_preds_scaled))
            train_true_real = self.scaler_y.inverse_transform(np.array(train_true_scaled))
            val_preds_real = self.scaler_y.inverse_transform(np.array(val_preds_scaled))
            val_true_real = self.scaler_y.inverse_transform(np.array(val_true_scaled))
           

           

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"epoch [{epoch+1}/{self.epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | ")

        # -------- Plot ----------
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

       # -------- Evaluation ----------
        residuals = val_true_real - val_preds_real

        mse = mean_squared_error(val_true_real, val_preds_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(val_true_real, val_preds_real)
        r2 = r2_score(val_true_real, val_preds_real)

        # Avoid division by zero in MAPE
        mape = np.mean(np.abs(residuals / np.clip(val_true_real, 1e-8, None))) * 100

        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)



        print("\n--- Validation Summary ---")
        print("\n" + "="*60)
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Mean Residual: {mean_residual:.4f}")
        print(f"  Std Residual:  {std_residual:.4f}")
        print("="*60)


        return {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "y_true": val_true_real,
            "y_pred": val_preds_real,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
    def predict(self, last_seq, n_steps=1):
         
            self.model.eval()
            seq = self.scaler_X.transform(last_seq)  # scale input
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            preds = []

            with torch.no_grad():
                for _ in range(n_steps):
                    pred = self.model(seq)  # (1, 1)
                    preds.append(pred.cpu().numpy().ravel()[0])

                    # create next step: keep last features, replace target with predicted
                    last_features = seq[:, -1, :].cpu().numpy()
                    next_step = last_features.copy()
                    next_step[0, 0] = pred.cpu().numpy().ravel()[0]  # replace Close price
                    seq = torch.cat([seq[:, 1:, :], torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).to(DEVICE)], dim=1)

            preds_real = self.scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()

            return preds_real






class TimeSeriesTrainerSimple2:
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
        self.scaler_y = MinMaxScaler()  # scale y now


    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_len):
            X_seq.append(X[i:i+self.seq_len])
            y_seq.append(y[i+self.seq_len])
        return np.array(X_seq), np.array(y_seq)
    


    

    def train(self, data):
        # ---------------- Prepare data ----------------
        X = data
        y = data[:, 0].reshape(-1, 1)

        # Split train/val BEFORE scaling
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]

        # Scale X and y
        X_train = self.scaler_X.fit_transform(X_train_raw)
        X_val = self.scaler_X.transform(X_val_raw)
        y_train = self.scaler_y.fit_transform(y_train_raw)
        y_val = self.scaler_y.transform(y_val_raw)

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)

        # DataLoaders
        train_loader = DataLoader(StockDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                               torch.tensor(y_train_seq, dtype=torch.float32)),
                                  batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(StockDataset(torch.tensor(X_val_seq, dtype=torch.float32),
                                             torch.tensor(y_val_seq, dtype=torch.float32)),
                                batch_size=self.batch_size, shuffle=False)

        self.model.to(DEVICE)

        for epoch in range(self.epochs):
            self.model.train()
            train_epoch_loss = []
            train_preds_scaled = []
            train_true_scaled = []

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())

                train_preds_scaled.extend(pred.detach().cpu().numpy())
                train_true_scaled.extend(yb.detach().cpu().numpy())

            avg_train_loss = np.mean(train_epoch_loss)
            self.train_losses.append(avg_train_loss)

            # ---------- Validation ----------
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
                    print(f"\n⏹ Early stopping at epoch {epoch+1}")
                    break

            # ---------- Epoch logging ----------
            train_preds_real = self.scaler_y.inverse_transform(np.array(train_preds_scaled))
            train_true_real = self.scaler_y.inverse_transform(np.array(train_true_scaled))
            val_preds_real = self.scaler_y.inverse_transform(np.array(val_preds_scaled))
            val_true_real = self.scaler_y.inverse_transform(np.array(val_true_scaled))

           

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"epoch [{epoch+1}/{self.epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                )
        # -------- Plot Loss ----------
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

        # -------- Evaluation ----------
        residuals = val_true_real - val_preds_real

        mse = mean_squared_error(val_true_real, val_preds_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(val_true_real, val_preds_real)
        r2 = r2_score(val_true_real, val_preds_real)
        mape = np.mean(np.abs(residuals / np.clip(val_true_real, 1e-8, None))) * 100

        print("\n--- Validation Summary ---")
        print("\n" + "="*60)
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Mean Residual: {np.mean(residuals):.4f}")
        print(f"  Std Residual:  {np.std(residuals):.4f}")
        print("="*60)

        return {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "y_true": val_true_real,
            "y_pred": val_preds_real,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }


    def predict(self, last_seq, n_steps=1):
        self.model.eval()
        seq = self.scaler_X.transform(last_seq)  # scale X only
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        preds = []

        with torch.no_grad():
            for _ in range(n_steps):
                pred = self.model(seq)
                preds.append(pred.cpu().numpy().ravel()[0])

                # create next step
                last_features = seq[:, -1, :].cpu().numpy()
                next_step = last_features.copy()
                next_step[0, 0] = pred.cpu().numpy().ravel()[0]
                seq = torch.cat([seq[:, 1:, :], torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).to(DEVICE)], dim=1)

        # scale y back
        preds_real = self.scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
        return preds_real


    # ----- Plot forecast helper -----
    def plot_forecast(self, y_true, y_pred, dates=None, model_name="Model",
                      zoom_last_n=120, figsize=(16, 10)):

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]

        if dates is None:
            dates = np.arange(n)
        else:
            dates = dates[-n:]

        residuals = y_true - y_pred
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(residuals / np.clip(y_true, 1e-8, None))) * 100

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Full forecast
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, y_true, label='Actual', linewidth=2, color='black', alpha=0.8)
        ax1.plot(dates, y_pred, label='Predicted', linewidth=1.5, linestyle='--', color='red', alpha=0.7)
        ax1.fill_between(dates, y_true, y_pred, alpha=0.2, color='red')
        ax1.set_title(f'{model_name} - Full Forecast | MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%',
                      fontweight='bold', fontsize=12)
        ax1.set_ylabel('Close Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Zoomed view
        ax2 = fig.add_subplot(gs[1, :])
        zoom_n = min(zoom_last_n, n)
        ax2.plot(dates[-zoom_n:], y_true[-zoom_n:], label='Actual',
                 linewidth=2, color='black', marker='o', markersize=3)
        ax2.plot(dates[-zoom_n:], y_pred[-zoom_n:], label='Predicted',
                 linewidth=1.5, linestyle='--', color='red', marker='s', markersize=3)
        ax2.set_title(f'Zoomed View - Last {zoom_n} Points', fontweight='bold')
        ax2.set_ylabel('Close Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Residuals
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.bar(range(n), residuals, alpha=0.6, color='steelblue', width=1.0)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Residuals (Prediction Errors)', fontweight='bold')
        ax3.set_ylabel('Residual')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Actual vs Predicted
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax4.set_xlabel('Actual')
        ax4.set_ylabel('Predicted')
        ax4.set_title('Actual vs Predicted', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary
        print(f"\n{'='*60}")
        print(f"  {model_name} - Performance Summary")
        print(f"{'='*60}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Mean Residual: {np.mean(residuals):.4f}")
        print(f"  Std Residual:  {np.std(residuals):.4f}")
        print(f"{'='*60}\n")

        # Interpretation
        if mape < 5:
            quality = "Excellent"
        elif mape < 10:
            quality = "Very Good"
        elif mape < 15:
            quality = "Good"
        elif mape < 25:
            quality = "Moderate"
        else:
            quality = "Needs Improvement"
        print(f"  Model Quality: {quality}")

        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Mean Residual": np.mean(residuals), "Std Residual": np.std(residuals)}
class TimeSeriesTrainerSimple3:
    def __init__(self, model, seq_len=20, batch_size=32, epochs=50, lr=0.001,
                 criterion=None, optimizer=None, early_stopping=True, patience=20):
        self.model = model
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.criterion = criterion if criterion else torch.nn.MSELoss()
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=lr)

        self.train_losses = []
        self.val_losses = []
        self.early_stopping = EarlyStopping(patience=patience) if early_stopping else None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_len):
            X_seq.append(X[i:i+self.seq_len])
            y_seq.append(y[i+self.seq_len])
        return np.array(X_seq), np.array(y_seq)

    def direction_accuracy(self, y_true, y_pred):
        # Up/Down accuracy
        return np.mean(np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])) * 100

    def train(self, data):
        # ---------------- Prepare data ----------------
        X = data
        y = data[:, 0].reshape(-1, 1)

        # Split train/val BEFORE scaling to avoid leakage
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]

        # Scale data
        X_train = self.scaler_X.fit_transform(X_train_raw)
        X_val = self.scaler_X.transform(X_val_raw)
        y_train = self.scaler_y.fit_transform(y_train_raw)
        y_val = self.scaler_y.transform(y_val_raw)

        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)

        train_loader = DataLoader(StockDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                               torch.tensor(y_train_seq, dtype=torch.float32)),
                                  batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(StockDataset(torch.tensor(X_val_seq, dtype=torch.float32),
                                             torch.tensor(y_val_seq, dtype=torch.float32)),
                                batch_size=self.batch_size, shuffle=False)

        self.model.to(DEVICE)

        for epoch in range(self.epochs):
            self.model.train()
            train_epoch_loss = []
            train_preds_scaled = []
            train_true_scaled = []

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
                train_preds_scaled.extend(pred.detach().cpu().numpy())
                train_true_scaled.extend(yb.detach().cpu().numpy())

            avg_train_loss = np.mean(train_epoch_loss)
            self.train_losses.append(avg_train_loss)

            # ---------- Validation ----------
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
                    print(f"\n⏹ Early stopping at epoch {epoch+1}")
                    break

            # ---------- Epoch logging ----------
            train_preds_real = self.scaler_y.inverse_transform(np.array(train_preds_scaled))
            train_true_real = self.scaler_y.inverse_transform(np.array(train_true_scaled))
            val_preds_real = self.scaler_y.inverse_transform(np.array(val_preds_scaled))
            val_true_real = self.scaler_y.inverse_transform(np.array(val_true_scaled))

            train_dir_acc = self.direction_accuracy(train_true_real, train_preds_real)
            val_dir_acc = self.direction_accuracy(val_true_real, val_preds_real)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch [{epoch+1}/{self.epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | ")

        # -------- Plot ----------
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

        # -------- Evaluation ----------
        mse = mean_squared_error(val_true_real, val_preds_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(val_true_real, val_preds_real)
        r2 = r2_score(val_true_real, val_preds_real)
        dir_acc = self.direction_accuracy(val_true_real, val_preds_real)

        print("\n--- Validation Summary ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R²  : {r2:.4f}")

        return {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "direction_accuracy": dir_acc,
            "y_true": val_true_real,
            "y_pred": val_preds_real,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }

    def predict(self, last_seq, n_steps=1):
        self.model.eval()
        seq = self.scaler_X.transform(last_seq)  # scale input
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        preds = []

        with torch.no_grad():
            for _ in range(n_steps):
                pred = self.model(seq)  # (1, 1)
                preds.append(pred.cpu().numpy().ravel()[0])

                # create next step: keep last features, replace target with predicted
                last_features = seq[:, -1, :].cpu().numpy()
                next_step = last_features.copy()
                next_step[0, 0] = pred.cpu().numpy().ravel()[0]  # replace Close price
                seq = torch.cat([seq[:, 1:, :], torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).to(DEVICE)], dim=1)

        preds_real = self.scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
        return preds_real
def set_seed(seed=62):
 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

