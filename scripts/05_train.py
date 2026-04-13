"""
05_train.py
-----------
Trains three models in sequence:

  Model 1 — Logistic Regression (sklearn)
    Input : 10 hand-crafted spatial features from data/features.csv
    Tuning: L2 C parameter via 5-fold CV on training set
    Output: models/logreg.pkl + models/logreg_results.json

  Model 2 — Custom CNN (PyTorch)
    Input : 64×64 float32 .npy arrays (1 channel)
    Arch  : 3× (Conv2d→BN→ReLU→MaxPool) → FC(256) → Dropout(0.5) → FC(4)
    Train : Adam lr=1e-3, cross-entropy, early stopping patience=10
    Output: models/cnn_best.pt + models/cnn_history.json

  Model 3 — ResNet-18 fine-tuned (PyTorch)
    Input : 64×64 arrays replicated to 3 channels
    Arch  : ImageNet-pretrained ResNet-18, fc replaced with Linear(512,4)
    Train : Adam lr=1e-4, cross-entropy, early stopping patience=10
    Output: models/resnet_best.pt + models/resnet_history.json

Augmentation (training only):
    RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5),
    RandomRotation(±15°), Additive Gaussian noise (σ=0.01)

Requirements:
    pip install torch torchvision scikit-learn pandas numpy

Usage:
    python scripts/05_train.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
FEAT_PATH  = ROOT / "data" / "features.csv"
SPLIT_PATH = ROOT / "data" / "splits.json"
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEED    = 42
EPOCHS  = 100
PATIENCE = 10
TIER_ORDER = ["Low", "Lower-Middle", "Upper-Middle", "High"]
TIER2IDX   = {t: i for i, t in enumerate(TIER_ORDER)}

for path in [FEAT_PATH, SPLIT_PATH]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run 04_split.py first.")

feat_df    = pd.read_csv(FEAT_PATH)
feat_index = feat_df.set_index("iso3")

with open(SPLIT_PATH) as f:
    splits = json.load(f)

FEATURE_COLS = [
    "mean_brightness", "total_light", "p90_brightness",
    "lit_fraction", "light_density",
    "spatial_dispersion", "hotspot_count",
    "gini", "entropy", "urban_concentration",
]


def get_label(iso3):
    return TIER2IDX[feat_index.loc[iso3, "income_tier"]]


# ===========================================================================
# MODEL 1 — Logistic Regression
# ===========================================================================

print("=" * 60)
print("MODEL 1: Logistic Regression")
print("=" * 60)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

def get_X_y(iso_list):
    X = feat_index.loc[iso_list, FEATURE_COLS].values.astype(np.float32)
    y = np.array([get_label(c) for c in iso_list])
    return X, y

X_train, y_train = get_X_y(splits["train"])
X_val,   y_val   = get_X_y(splits["val"])
X_test,  y_test  = get_X_y(splits["test"])

scaler  = StandardScaler().fit(X_train)
Xtr_s   = scaler.transform(X_train)
Xva_s   = scaler.transform(X_val)
Xte_s   = scaler.transform(X_test)

logreg = LogisticRegressionCV(
    Cs=10, cv=5, penalty="l2", max_iter=1000,
    multi_class="multinomial", solver="lbfgs",
    random_state=SEED, n_jobs=-1,
)
logreg.fit(Xtr_s, y_train)

val_preds  = logreg.predict(Xva_s)
test_preds = logreg.predict(Xte_s)

val_acc  = accuracy_score(y_val,  val_preds)
val_f1   = f1_score(y_val,  val_preds, average="macro", zero_division=0)
test_acc = accuracy_score(y_test, test_preds)
test_f1  = f1_score(y_test, test_preds, average="macro", zero_division=0)

print(f"  Best C : {logreg.C_[0]:.4f}")
print(f"  Val    : acc={val_acc:.4f}  macro-F1={val_f1:.4f}")
print(f"  Test   : acc={test_acc:.4f}  macro-F1={test_f1:.4f}")
print(classification_report(y_test, test_preds, target_names=TIER_ORDER, zero_division=0))

logreg_results = {
    "best_C": float(logreg.C_[0]),
    "val_acc": val_acc, "val_macro_f1": val_f1,
    "test_acc": test_acc, "test_macro_f1": test_f1,
    "test_preds": test_preds.tolist(),
    "test_true":  y_test.tolist(),
}
with open(MODEL_DIR / "logreg_results.json", "w") as f:
    json.dump(logreg_results, f, indent=2)

with open(MODEL_DIR / "logreg.pkl", "wb") as f:
    pickle.dump({"model": logreg, "scaler": scaler}, f)

print(f"  Saved → models/logreg.pkl + models/logreg_results.json\n")


# ===========================================================================
# PyTorch setup
# ===========================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.models as tv_models

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"PyTorch device: {DEVICE}\n")

BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Dataset with optional augmentation
# ---------------------------------------------------------------------------

class NTLDataset(Dataset):
    def __init__(self, iso_list, augment=False, three_channel=False):
        self.items        = [(iso3, get_label(iso3)) for iso3 in iso_list]
        self.augment      = augment
        self.three_channel = three_channel
        self._rng         = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        iso3, label = self.items[idx]
        arr = np.load(PROC_DIR / f"{iso3}_2022.npy").astype(np.float32)
        img = torch.from_numpy(arr).unsqueeze(0)   # (1, 64, 64)

        if self.augment:
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
            if torch.rand(1) > 0.5:
                img = TF.vflip(img)
            angle = float(torch.empty(1).uniform_(-15, 15))
            img   = TF.rotate(img, angle)
            img   = img + torch.randn_like(img) * 0.01
            img   = img.clamp(0.0, 1.0)

        if self.three_channel:
            img = img.repeat(3, 1, 1)   # (3, 64, 64)

        return img, label


def make_loaders(three_channel=False):
    tr = DataLoader(NTLDataset(splits["train"], augment=True,  three_channel=three_channel),
                    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va = DataLoader(NTLDataset(splits["val"],   augment=False, three_channel=three_channel),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te = DataLoader(NTLDataset(splits["test"],  augment=False, three_channel=three_channel),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr, va, te


# ---------------------------------------------------------------------------
# Training loop with early stopping
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, optimizer, name):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(imgs)
        train_loss = running_loss / len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss, all_preds, all_true = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                val_loss += criterion(out, labels).item() * len(imgs)
                all_preds.extend(out.argmax(1).cpu().tolist())
                all_true.extend(labels.cpu().tolist())
        val_loss /= len(val_loader.dataset)
        val_acc   = accuracy_score(all_true, all_preds)
        val_f1    = f1_score(all_true, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{name}] Epoch {epoch:3d} | "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

        # early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), MODEL_DIR / f"{name}_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  [{name}] Early stop at epoch {epoch}")
                break

    return history


def eval_on_test(model, test_loader, name):
    model.load_state_dict(torch.load(MODEL_DIR / f"{name}_best.pt", map_location=DEVICE))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            all_preds.extend(model(imgs).argmax(1).cpu().tolist())
            all_true.extend(labels.tolist())
    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
    print(f"  [{name}] TEST  acc={acc:.4f}  macro-F1={f1:.4f}")
    print(classification_report(all_true, all_preds, target_names=TIER_ORDER, zero_division=0))
    return acc, f1, all_preds, all_true


# ===========================================================================
# MODEL 2 — Custom CNN
# ===========================================================================

print("=" * 60)
print("MODEL 2: Custom CNN")
print("=" * 60)


class SmallCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 → 32×32
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2 → 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3 → 8×8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


cnn_model = SmallCNN().to(DEVICE)
cnn_opt   = optim.Adam(cnn_model.parameters(), lr=1e-3)
tr_cnn, va_cnn, te_cnn = make_loaders(three_channel=False)

cnn_history = train_model(cnn_model, tr_cnn, va_cnn, cnn_opt, "cnn")
cnn_acc, cnn_f1, cnn_preds, cnn_true = eval_on_test(cnn_model, te_cnn, "cnn")

cnn_results = {
    "test_acc": cnn_acc, "test_macro_f1": cnn_f1,
    "test_preds": cnn_preds, "test_true": cnn_true,
}
with open(MODEL_DIR / "cnn_history.json", "w") as f:
    json.dump({**cnn_history, **cnn_results}, f, indent=2)
print(f"  Saved → models/cnn_best.pt + models/cnn_history.json\n")


# ===========================================================================
# MODEL 3 — ResNet-18 fine-tuned
# ===========================================================================

print("=" * 60)
print("MODEL 3: ResNet-18 (fine-tuned)")
print("=" * 60)

resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Linear(512, 4)
resnet     = resnet.to(DEVICE)

resnet_opt = optim.Adam(resnet.parameters(), lr=1e-4)
tr_rn, va_rn, te_rn = make_loaders(three_channel=True)

resnet_history = train_model(resnet, tr_rn, va_rn, resnet_opt, "resnet")
rn_acc, rn_f1, rn_preds, rn_true = eval_on_test(resnet, te_rn, "resnet")

resnet_results = {
    "test_acc": rn_acc, "test_macro_f1": rn_f1,
    "test_preds": rn_preds, "test_true": rn_true,
}
with open(MODEL_DIR / "resnet_history.json", "w") as f:
    json.dump({**resnet_history, **resnet_results}, f, indent=2)
print(f"  Saved → models/resnet_best.pt + models/resnet_history.json\n")


# ===========================================================================
# Summary
# ===========================================================================

print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"  {'Model':<20} {'Test Acc':>10}  {'Macro-F1':>10}")
print(f"  {'-'*44}")
print(f"  {'LogReg':<20} {logreg_results['test_acc']:>10.4f}  {logreg_results['test_macro_f1']:>10.4f}")
print(f"  {'CNN':<20} {cnn_acc:>10.4f}  {cnn_f1:>10.4f}")
print(f"  {'ResNet-18':<20} {rn_acc:>10.4f}  {rn_f1:>10.4f}")
print("=" * 60)
print("Next: python scripts/06_evaluate.py")
