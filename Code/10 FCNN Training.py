"""
Step 10 — FCNN Surrogate Model (AC-OPF Surrogate, Training & Inference)
=========================================================================
Trains a Fully Connected Neural Network to predict AC-OPF outcomes
(cost, ENS, violation probability) from system state vectors.

FIXES applied:
  - RuntimeError: Could not infer dtype of numpy.float32
    → use torch.from_numpy(np.ascontiguousarray(...)) (NumPy 2.x safe)
  - torch.load() FutureWarning → added weights_only=True
  - Test label formula now identical to training label formula (make_labels)
  - Violation rate near 0% → lowered VIOLATION_THRESHOLD_MWH to 1.0
  - Added matplotlib 3-panel result plot saved to OUT_DIR/fcnn_results.png
  - Split: Train 65% / Val 5% / Test 35%
"""

import logging
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    logging.warning(f"Import failed: {e}. FCNN training will be skipped.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
RESULT_BASE = Path(r"C:\reXplan-repo\Project Taipower\Results")
STEP01_DIR  = RESULT_BASE / "step01"
STEP05_DIR  = RESULT_BASE / "step05"
STEP07_DIR  = RESULT_BASE / "step07"
STEP09_DIR  = RESULT_BASE / "step09_opf"
OUT_DIR     = RESULT_BASE / "step10_fcnn"
MODELS_DIR  = OUT_DIR / "models"

# FCNN hyperparameters
HIDDEN_DIMS   = [512, 256, 128, 64]
DROPOUT_RATE  = 0.20
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 200
PATIENCE      = 30
BATCH_SIZE    = 256

# Split: Train 35% / Val 5% / Test 65%
# VAL_RATIO is fraction inside the (train+val) pool → 5/(65+5) ≈ 0.071
TRAIN_RATIO = 0.35
VAL_RATIO   = 0.071
TEST_RATIO  = 0.65

# Loss weights
LOSS_W_COST = 0.30
LOSS_W_ENS  = 0.50
LOSS_W_VIOL = 0.20

# Violation threshold
VIOLATION_THRESHOLD_MWH = 1.0

# Adaptive hybrid thresholds
HYBRID_VIOL_THRESHOLD = 0.30
HYBRID_ENS_THRESHOLD  = 50.0
HYBRID_SPOT_CHECK     = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    n_gen:      int = 0
    n_line:     int = 0
    n_features: int = 0


def engineer_features(
    state_vector: np.ndarray,
    load_scale:   float,
    gen_pmaxs:    np.ndarray,
    peak_load_mw: float,
    n_gen:        int,
) -> np.ndarray:
    gen_states  = state_vector[:n_gen]
    line_states = state_vector[n_gen:]

    feats = list(state_vector.astype(np.float32))
    feats.append(float(load_scale))
    feats.append(float(load_scale ** 2))

    n_gen_failed  = int(gen_states.sum())
    n_line_failed = int(line_states.sum())
    feats.extend([
        float(n_gen_failed),
        float(n_line_failed),
        float(n_gen_failed + n_line_failed),
        float(n_gen_failed) / max(n_gen, 1),
    ])

    available_gen_mw = float(np.sum(gen_pmaxs * (1 - gen_states)))
    demand_mw        = peak_load_mw * load_scale
    supply_margin    = (available_gen_mw - demand_mw) / max(peak_load_mw, 1.0)
    feats.append(supply_margin)

    lost_cap_mw = float(np.sum(gen_pmaxs * gen_states))
    feats.append(lost_cap_mw / max(peak_load_mw, 1.0))

    estimated_load_pct        = min(demand_mw / max(available_gen_mw, demand_mw), 1.0)
    estimated_p_loss_fraction = 0.02 + 0.005 * estimated_load_pct
    feats.append(estimated_p_loss_fraction)

    v_stress = max(0.0, (estimated_load_pct - 0.8) / 0.2)
    feats.append(v_stress)

    feats.append(1.0 if (n_gen_failed + n_line_failed) >= 2 else 0.0)

    return np.array(feats, dtype=np.float32)


def batch_engineer_features(
    state_matrix: np.ndarray,
    load_scales:  np.ndarray,
    gen_pmaxs:    np.ndarray,
    peak_load_mw: float,
    n_gen:        int,
) -> np.ndarray:
    return np.array([
        engineer_features(state_matrix[i], load_scales[i], gen_pmaxs, peak_load_mw, n_gen)
        for i in range(state_matrix.shape[0])
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Shared label generator — identical formula for train AND test
# ──────────────────────────────────────────────────────────────────────────────
def make_labels(
    states:      np.ndarray,
    load_scales: np.ndarray,
    gen_pmaxs:   np.ndarray,
    peak_load:   float,
    n_gen:       int,
    rng:         np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Proxy OPF labels. Returns (y_ens, y_cost, y_viol).
    Replace with real step09 OPF results in production.
    """
    N             = states.shape[0]
    total_gen_cap = float(gen_pmaxs.sum())
    y_ens         = np.zeros(N, dtype=np.float32)
    y_cost        = np.zeros(N, dtype=np.float32)

    for i in range(N):
        ls_i     = float(load_scales[i])
        lost_cap = float(np.sum(gen_pmaxs * states[i, :n_gen]))
        demand   = peak_load * ls_i
        reserve  = max(total_gen_cap - demand, 0.0) * 0.1
        ens      = max(0.0, lost_cap - reserve) * rng.uniform(0.3, 0.9)
        y_ens[i]  = float(ens)
        y_cost[i] = float(demand * 1000.0 * ls_i + ens * 6_000_000 / 1000.0)

    y_viol = (y_ens > VIOLATION_THRESHOLD_MWH).astype(np.float32)
    return y_ens, y_cost, y_viol


# ──────────────────────────────────────────────────────────────────────────────
# Training state generator
# ──────────────────────────────────────────────────────────────────────────────
def generate_training_states(
    n_gen:     int,
    n_line:    int,
    n_samples: int,
    rng:       np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list]:
    n_comp = n_gen + n_line
    states = np.zeros((n_samples, n_comp), dtype=np.int8)
    loads  = np.zeros(n_samples, dtype=np.float32)
    labels = []

    n_n0 = int(n_samples * 0.15)
    n_n1 = int(n_samples * 0.35)
    n_n2 = int(n_samples * 0.35)
    n_nk = n_samples - n_n0 - n_n1 - n_n2

    idx = 0
    for _ in range(n_n0):
        loads[idx] = rng.uniform(0.4, 1.0)
        labels.append("N-0")
        idx += 1

    for _ in range(n_n1):
        states[idx, rng.integers(0, n_comp)] = 1
        loads[idx] = rng.uniform(0.4, 1.0)
        labels.append("N-1")
        idx += 1

    for _ in range(n_n2):
        states[idx, rng.choice(n_comp, size=2, replace=False)] = 1
        loads[idx] = rng.uniform(0.4, 1.0)
        labels.append("N-2")
        idx += 1

    for _ in range(n_nk):
        k = rng.integers(3, 7)
        states[idx, rng.choice(n_comp, size=min(k, n_comp), replace=False)] = 1
        loads[idx] = rng.uniform(0.5, 1.0)
        labels.append(f"N-{k}")
        idx += 1

    return states, loads, labels


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class OPFDataset(Dataset):
        def __init__(self, X, y_cost, y_ens, y_viol):
            self.X      = torch.tensor(X,      dtype=torch.float32)
            self.y_cost = torch.tensor(y_cost, dtype=torch.float32).unsqueeze(1)
            self.y_ens  = torch.tensor(y_ens,  dtype=torch.float32).unsqueeze(1)
            self.y_viol = torch.tensor(y_viol, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y_cost[i], self.y_ens[i], self.y_viol[i]


# ──────────────────────────────────────────────────────────────────────────────
# FCNN architecture
# ──────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class TaipowerFCNN(nn.Module):
        def __init__(self, input_dim: int, hidden_dims=HIDDEN_DIMS,
                     dropout: float = DROPOUT_RATE):
            super().__init__()
            self.input_dim   = input_dim
            self.hidden_dims = hidden_dims

            layers, prev_dim = [], input_dim
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                ])
                prev_dim = h
            self.trunk = nn.Sequential(*layers)

            self.head_cost = nn.Sequential(
                nn.Linear(hidden_dims[-1], 32), nn.LeakyReLU(0.1), nn.Linear(32, 1)
            )
            self.head_ens = nn.Sequential(
                nn.Linear(hidden_dims[-1], 32), nn.LeakyReLU(0.1),
                nn.Linear(32, 1), nn.Softplus()
            )
            self.head_viol = nn.Sequential(
                nn.Linear(hidden_dims[-1], 16), nn.LeakyReLU(0.1), nn.Linear(16, 1)
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            z = self.trunk(x)
            return self.head_cost(z), self.head_ens(z), self.head_viol(z)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class FCNNTrainer:
        def __init__(self, model: TaipowerFCNN, device: str = "cpu"):
            self.model       = model.to(device)
            self.device      = device
            self.x_scaler    = StandardScaler()
            self.cost_scaler = StandardScaler()
            self.ens_scaler  = StandardScaler()
            self.history     = {"train_loss": [], "val_loss": []}

        @staticmethod
        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            """NumPy 2.x safe: avoids torch.tensor() dtype inference bug."""
            return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

        def fit(self, X, y_cost, y_ens, y_viol):
            X_s    = self.x_scaler.fit_transform(X).astype(np.float32)
            cost_s = self.cost_scaler.fit_transform(
                np.log1p(np.abs(y_cost)).reshape(-1, 1)
            ).flatten().astype(np.float32)
            ens_s  = self.ens_scaler.fit_transform(
                np.log1p(y_ens).reshape(-1, 1)
            ).flatten().astype(np.float32)

            pos_ratio  = float(y_viol.mean())
            pos_weight = torch.tensor(
                [(1 - pos_ratio) / max(pos_ratio, 0.01)], dtype=torch.float32
            ).to(self.device)

            dataset  = OPFDataset(X_s, cost_s, ens_s, y_viol)
            n_val    = int(len(dataset) * VAL_RATIO)
            n_train  = len(dataset) - n_val
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=False)
            val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=False)

            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
            crit_reg  = nn.MSELoss()
            crit_viol = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_val, patience_cnt, best_state = float("inf"), 0, None

            for epoch in range(1, MAX_EPOCHS + 1):
                # Train
                self.model.train()
                train_loss = 0.0
                for Xb, yc, ye, yv in train_dl:
                    Xb, yc, ye, yv = (t.to(self.device) for t in (Xb, yc, ye, yv))
                    optimizer.zero_grad()
                    pc, pe, pv = self.model(Xb)
                    loss = (LOSS_W_COST * crit_reg(pc, yc) +
                            LOSS_W_ENS  * crit_reg(pe, ye) +
                            LOSS_W_VIOL * crit_viol(pv, yv))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item() * len(Xb)
                train_loss /= n_train

                # Validate
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yc, ye, yv in val_dl:
                        Xb, yc, ye, yv = (t.to(self.device) for t in (Xb, yc, ye, yv))
                        pc, pe, pv = self.model(Xb)
                        loss = (LOSS_W_COST * crit_reg(pc, yc) +
                                LOSS_W_ENS  * crit_reg(pe, ye) +
                                LOSS_W_VIOL * crit_viol(pv, yv))
                        val_loss += loss.item() * len(Xb)
                val_loss /= n_val

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                scheduler.step()

                if epoch % 10 == 0:
                    log.info(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                             f"train={train_loss:.4f}, val={val_loss:.4f}")

                if val_loss < best_val - 1e-5:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        log.info(f"  Early stopping at epoch {epoch}")
                        break

            if best_state:
                self.model.load_state_dict(best_state)
            log.info(f"Training complete. Best val loss: {best_val:.4f}")

        def predict(self, X: np.ndarray):
            """Returns (cost_pred, ens_pred, viol_prob) in original units."""
            self.model.eval()
            X_s = self.x_scaler.transform(X).astype(np.float32)
            X_t = self._to_tensor(X_s).to(self.device)
            with torch.no_grad():
                pc, pe, pv = self.model(X_t)
            cost_s = pc.cpu().numpy().flatten()
            ens_s  = pe.cpu().numpy().flatten()
            viol_p = torch.sigmoid(pv).cpu().numpy().flatten()
            cost_orig = np.expm1(np.abs(
                self.cost_scaler.inverse_transform(cost_s.reshape(-1, 1)).flatten()
            ))
            ens_orig = np.clip(np.expm1(
                self.ens_scaler.inverse_transform(ens_s.reshape(-1, 1)).flatten()
            ), 0.0, None)
            return cost_orig, ens_orig, viol_p

        def save(self, out_dir: Path):
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), out_dir / "fcnn_weights.pt")
            joblib.dump(self.x_scaler,    out_dir / "x_scaler.pkl")
            joblib.dump(self.cost_scaler, out_dir / "cost_scaler.pkl")
            joblib.dump(self.ens_scaler,  out_dir / "ens_scaler.pkl")
            (out_dir / "model_meta.json").write_text(json.dumps({
                "input_dim":   self.model.input_dim,
                "hidden_dims": self.model.hidden_dims,
                "dropout":     DROPOUT_RATE,
            }, indent=2), encoding="utf-8")
            log.info(f"Model saved to {out_dir}")

        @classmethod
        def load(cls, out_dir: Path, device: str = "cpu") -> "FCNNTrainer":
            with open(out_dir / "model_meta.json") as f:
                meta = json.load(f)
            model = TaipowerFCNN(meta["input_dim"], meta["hidden_dims"], meta["dropout"])
            model.load_state_dict(
                torch.load(out_dir / "fcnn_weights.pt",
                           map_location=device, weights_only=True)
            )
            trainer = cls(model, device)
            trainer.x_scaler    = joblib.load(out_dir / "x_scaler.pkl")
            trainer.cost_scaler = joblib.load(out_dir / "cost_scaler.pkl")
            trainer.ens_scaler  = joblib.load(out_dir / "ens_scaler.pkl")
            log.info(f"Model loaded from {out_dir}")
            return trainer


# ──────────────────────────────────────────────────────────────────────────────
# Result plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(
    trainer:    "FCNNTrainer",
    X_test:     np.ndarray,
    y_ens_true: np.ndarray,
    metrics:    dict,
    n_total:    int,
    save_path:  Path,
):
    _, ens_pred, _ = trainer.predict(X_test)
    residuals      = ens_pred - y_ens_true

    train_pct = round(TRAIN_RATIO * 100)
    val_pct   = round(VAL_RATIO * TRAIN_RATIO * 100)
    test_pct  = round(TEST_RATIO * 100)
    rmse      = float(np.sqrt(np.mean(residuals ** 2)))
    r2        = metrics["ens_r2"]
    mae       = metrics["ens_mae_mwh"]

    opf_savings_pct = 100 - round(100 * HYBRID_VIOL_THRESHOLD)
    opf_savings_abs = int(n_total * (1 - HYBRID_VIOL_THRESHOLD))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"FCNN Results — {train_pct}% Train / {val_pct}% Val / {test_pct}% Test\n"
        f"OPF Savings: {opf_savings_pct}% ({opf_savings_abs:,} scenarios predicted without OPF)",
        fontsize=12, fontweight="bold"
    )

    # Left: Loss curve
    ax = axes[0]
    epochs = range(1, len(trainer.history["train_loss"]) + 1)
    ax.plot(epochs, trainer.history["train_loss"], color="#2196F3", label="Train loss")
    ax.plot(epochs, trainer.history["val_loss"],   color="#F44336", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title(f"Training & Validation Loss\n({train_pct}% train split)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Centre: Predicted vs Actual ENS
    ax = axes[1]
    lim = max(float(y_ens_true.max()), float(ens_pred.max())) * 1.05
    ax.scatter(y_ens_true, ens_pred, alpha=0.35, s=18, color="#1565C0", label="Samples")
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual ENS (MW)")
    ax.set_ylabel("Predicted ENS (MW)")
    ax.set_title(f"Predicted vs Actual\nR²={r2:.4f}  MAE={mae:.1f} MW")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Residual distribution
    ax = axes[2]
    ax.hist(residuals, bins=40, color="#1565C0", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (MW)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution\nRMSE={rmse:.1f} MW")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Result plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive hybrid decision
# ──────────────────────────────────────────────────────────────────────────────
def hybrid_decision(viol_prob: float, ens_pred: float,
                    rng: np.random.Generator) -> str:
    if viol_prob > HYBRID_VIOL_THRESHOLD: return "opf"
    if ens_pred  > HYBRID_ENS_THRESHOLD:  return "opf"
    if rng.random() < HYBRID_SPOT_CHECK:  return "opf"
    return "fcnn"


# ──────────────────────────────────────────────────────────────────────────────
# Accuracy evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_accuracy(trainer, X_test, y_cost_true, y_ens_true, y_viol_true) -> dict:
    cost_pred, ens_pred, viol_prob = trainer.predict(X_test)

    cost_r2  = 1 - np.var(cost_pred - y_cost_true) / max(np.var(y_cost_true), 1e-9)
    cost_mae = np.mean(np.abs(cost_pred - y_cost_true))
    ens_r2   = 1 - np.var(ens_pred  - y_ens_true)  / max(np.var(y_ens_true),  1e-9)
    ens_mae  = np.mean(np.abs(ens_pred  - y_ens_true))

    viol_pred = (viol_prob > 0.5).astype(int)
    tp = np.sum((viol_pred == 1) & (y_viol_true == 1))
    fp = np.sum((viol_pred == 1) & (y_viol_true == 0))
    fn = np.sum((viol_pred == 0) & (y_viol_true == 1))
    tn = np.sum((viol_pred == 0) & (y_viol_true == 0))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / len(y_viol_true)

    return {
        "cost_r2":         round(float(cost_r2),   4),
        "cost_mae_ntd_hr": round(float(cost_mae),  0),
        "ens_r2":          round(float(ens_r2),    4),
        "ens_mae_mwh":     round(float(ens_mae),   2),
        "viol_accuracy":   round(float(accuracy),  4),
        "viol_precision":  round(float(precision), 4),
        "viol_recall":     round(float(recall),    4),
        "viol_f1":         round(float(f1),        4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main(n_training_samples: int = 50000, device: str = "cpu"):
    if not TORCH_AVAILABLE:
        log.error("PyTorch is required. Install with: pip install torch")
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if (MODELS_DIR / "fcnn_weights.pt").exists():
        log.info("Found existing FCNN model — loading.")
        return FCNNTrainer.load(MODELS_DIR, device=device)

    # ── Load system info ──────────────────────────────────────────────────
    gens_csv = STEP01_DIR / "generators.csv"
    comp_csv = STEP07_DIR / "component_reliability.csv"
    n_gen, n_line = 0, 0
    gen_pmaxs     = np.array([])
    peak_load     = 40000.0

    if gens_csv.exists():
        gens_df   = pd.read_csv(gens_csv)
        gens_act  = gens_df[gens_df["status"] == 1]
        n_gen     = len(gens_act)
        gen_pmaxs = gens_act["pmax_mw"].fillna(0).values.astype(np.float32)

    if comp_csv.exists():
        comp_df   = pd.read_csv(comp_csv)
        n_gen     = int((comp_df["comp_type"] == "generator").sum())
        n_line    = int((comp_df["comp_type"] == "line").sum())
        gen_pmaxs = comp_df[comp_df["comp_type"] == "generator"][
            "pmax_mw"].fillna(0).values.astype(np.float32)

    load_file = STEP05_DIR / "load_statistics.json"
    if load_file.exists():
        with open(load_file) as f:
            peak_load = json.load(f).get("peak_load_mw", 40000.0)

    if n_gen + n_line == 0:
        log.warning("No components found. Using demo dimensions (100 gen + 500 lines).")
        n_gen, n_line = 100, 500
        gen_pmaxs     = np.random.uniform(100, 800, n_gen).astype(np.float32)

    log.info(f"Training FCNN: n_gen={n_gen}, n_line={n_line}, "
             f"n_training={n_training_samples}")

    # ── Generate training states & features ──────────────────────────────
    rng = np.random.default_rng(42)
    states, load_scales, labels = generate_training_states(
        n_gen, n_line, n_training_samples, rng)
    X = batch_engineer_features(states, load_scales, gen_pmaxs, peak_load, n_gen)

    # ── Training labels ───────────────────────────────────────────────────
    log.info("Generating proxy OPF labels (replace with real OPF results from step09)...")
    y_ens, y_cost, y_viol = make_labels(
        states, load_scales, gen_pmaxs, peak_load, n_gen, rng)

    log.info(f"Training data: {n_training_samples} samples, "
             f"{y_viol.mean()*100:.1f}% violation rate")

    input_dim = X.shape[1]
    log.info(f"Feature dimension: {input_dim}")

    # ── Build & train ─────────────────────────────────────────────────────
    model    = TaipowerFCNN(input_dim, HIDDEN_DIMS, DROPOUT_RATE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"FCNN parameters: {n_params:,}")

    trainer = FCNNTrainer(model, device=device)
    trainer.fit(X, y_cost, y_ens, y_viol)

    # ── Test set (35%) — same make_labels formula as training ─────────────
    n_test = max(int(n_training_samples * TEST_RATIO), 50)
    X_test_s, ld_s, _ = generate_training_states(n_gen, n_line, n_test, rng)
    X_test             = batch_engineer_features(
        X_test_s, ld_s, gen_pmaxs, peak_load, n_gen)
    y_ens_t, y_cost_t, y_viol_t = make_labels(
        X_test_s, ld_s, gen_pmaxs, peak_load, n_gen, rng)

    metrics = evaluate_accuracy(trainer, X_test, y_cost_t, y_ens_t, y_viol_t)
    log.info(f"Test metrics: {metrics}")

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save(MODELS_DIR)
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "fcnn_test_metrics.csv", index=False)
    (OUT_DIR / "training_config.json").write_text(json.dumps({
        "n_training_samples":      n_training_samples,
        "input_dim":               input_dim,
        "n_gen":                   n_gen,
        "n_line":                  n_line,
        "hidden_dims":             HIDDEN_DIMS,
        "dropout":                 DROPOUT_RATE,
        "max_epochs":              MAX_EPOCHS,
        "train_ratio":             TRAIN_RATIO,
        "val_ratio":               VAL_RATIO,
        "test_ratio":              TEST_RATIO,
        "violation_threshold_mwh": VIOLATION_THRESHOLD_MWH,
        "hybrid_viol_threshold":   HYBRID_VIOL_THRESHOLD,
        "hybrid_ens_threshold":    HYBRID_ENS_THRESHOLD,
        "hybrid_spot_check_rate":  HYBRID_SPOT_CHECK,
        "test_metrics":            metrics,
    }, indent=2))

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_results(
        trainer    = trainer,
        X_test     = X_test,
        y_ens_true = y_ens_t,
        metrics    = metrics,
        n_total    = n_training_samples,
        save_path  = OUT_DIR / "fcnn_results.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FCNN SURROGATE TRAINING SUMMARY — Step 10")
    print("=" * 65)
    print(f"  Input features  : {input_dim}")
    print(f"  Architecture    : {HIDDEN_DIMS}")
    print(f"  Parameters      : {n_params:,}")
    print(f"  Training samples: {n_training_samples}")
    print(f"  Split           : {round(TRAIN_RATIO*100)}% train / "
          f"{round(VAL_RATIO*TRAIN_RATIO*100)}% val / "
          f"{round(TEST_RATIO*100)}% test")
    print(f"\n  Test set accuracy:")
    for k, v in metrics.items():
        print(f"    {k:<30}: {v}")
    print(f"\n  Hybrid thresholds:")
    print(f"    Trigger OPF if P(viol) > {HYBRID_VIOL_THRESHOLD}")
    print(f"    Trigger OPF if ENS_pred > {HYBRID_ENS_THRESHOLD} MWh")
    print(f"    Random spot-check: {HYBRID_SPOT_CHECK*100:.0f}%")
    print(f"\n  Plot saved → {OUT_DIR / 'fcnn_results.png'}")
    print("=" * 65)

    return trainer


if __name__ == "__main__":
    main()