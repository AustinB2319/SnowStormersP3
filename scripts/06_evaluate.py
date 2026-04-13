"""
06_evaluate.py
--------------
Evaluates all three trained models on the held-out test set and produces
a comprehensive set of diagnostic figures saved to figures/.

Outputs
-------
  figures/training_curves.png      — loss + val-F1 curves for CNN & ResNet
  figures/confusion_matrices.png   — one confusion matrix per model
  figures/model_comparison.png     — accuracy & macro-F1 bar chart
  figures/gradcam_cnn.png          — Grad-CAM overlays (CNN, best model)
  figures/gradcam_resnet.png       — Grad-CAM overlays (ResNet-18)
  figures/feature_importance.png   — LogReg coefficient heatmap

Interpretation printed to console:
  - Which tier pairs are hardest to separate
  - Whether Lower-Middle/Upper-Middle boundary is unresolvable
  - Known outlier countries (SYR, PSE, PRK …)
  - Whether CNN adds signal over LogReg

Requirements:
    pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn

Usage:
    python scripts/06_evaluate.py
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as tv_models
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
FEAT_PATH  = ROOT / "data" / "features.csv"
SPLIT_PATH = ROOT / "data" / "splits.json"
MODEL_DIR  = ROOT / "models"
FIG_DIR    = ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)

for p in [FEAT_PATH, SPLIT_PATH]:
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Run 04_split.py first.")
for name in ["logreg.pkl", "logreg_results.json", "cnn_best.pt",
             "cnn_history.json", "resnet_best.pt", "resnet_history.json"]:
    if not (MODEL_DIR / name).exists():
        sys.exit(f"ERROR: models/{name} not found. Run 05_train.py first.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED       = 42
TIER_ORDER = ["Low", "Lower-Middle", "Upper-Middle", "High"]
TIER2IDX   = {t: i for i, t in enumerate(TIER_ORDER)}
TIER_COLORS = {
    "Low": "#d73027", "Lower-Middle": "#fc8d59",
    "Upper-Middle": "#91bfdb", "High": "#4575b4",
}
DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
FEATURE_COLS = [
    "mean_brightness", "total_light", "p90_brightness",
    "lit_fraction", "light_density",
    "spatial_dispersion", "hotspot_count",
    "gini", "entropy", "urban_concentration",
]

feat_df    = pd.read_csv(FEAT_PATH)
feat_index = feat_df.set_index("iso3")

with open(SPLIT_PATH) as f:
    splits = json.load(f)


def get_label(iso3):
    return TIER2IDX[feat_index.loc[iso3, "income_tier"]]


# ---------------------------------------------------------------------------
# Dataset (no augmentation)
# ---------------------------------------------------------------------------

class NTLDataset(Dataset):
    def __init__(self, iso_list, three_channel=False):
        self.items         = [(iso3, get_label(iso3)) for iso3 in iso_list]
        self.three_channel = three_channel

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        iso3, label = self.items[idx]
        arr = np.load(PROC_DIR / f"{iso3}_2022.npy").astype(np.float32)
        img = torch.from_numpy(arr).unsqueeze(0)
        if self.three_channel:
            img = img.repeat(3, 1, 1)
        return img, label, iso3


# ---------------------------------------------------------------------------
# Rebuild model architectures
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


cnn_model = SmallCNN().to(DEVICE)
cnn_model.load_state_dict(torch.load(MODEL_DIR / "cnn_best.pt", map_location=DEVICE))
cnn_model.eval()

resnet = tv_models.resnet18(weights=None)
resnet.fc = nn.Linear(512, 4)
resnet.load_state_dict(torch.load(MODEL_DIR / "resnet_best.pt", map_location=DEVICE))
resnet = resnet.to(DEVICE)
resnet.eval()

with open(MODEL_DIR / "logreg.pkl", "rb") as f:
    logreg_bundle = pickle.load(f)
logreg  = logreg_bundle["model"]
scaler  = logreg_bundle["scaler"]

with open(MODEL_DIR / "logreg_results.json")  as f: lr_res  = json.load(f)
with open(MODEL_DIR / "cnn_history.json")      as f: cnn_res = json.load(f)
with open(MODEL_DIR / "resnet_history.json")   as f: rn_res  = json.load(f)

# ---------------------------------------------------------------------------
# Collect test-set predictions for all models
# ---------------------------------------------------------------------------

test_iso   = splits["test"]
y_true     = np.array([get_label(c) for c in test_iso])
X_test     = feat_index.loc[test_iso, FEATURE_COLS].values.astype(np.float32)
X_test_s   = scaler.transform(X_test)
lr_preds   = logreg.predict(X_test_s)

# CNN preds
te_cnn  = DataLoader(NTLDataset(test_iso, three_channel=False), batch_size=16)
te_rn   = DataLoader(NTLDataset(test_iso, three_channel=True),  batch_size=16)

cnn_preds, rn_preds = [], []
with torch.no_grad():
    for imgs, _, _ in te_cnn:
        cnn_preds.extend(cnn_model(imgs.to(DEVICE)).argmax(1).cpu().tolist())
    for imgs, _, _ in te_rn:
        rn_preds.extend(resnet(imgs.to(DEVICE)).argmax(1).cpu().tolist())

cnn_preds = np.array(cnn_preds)
rn_preds  = np.array(rn_preds)

model_results = {
    "Logistic Regression": {"preds": lr_preds,  "acc": accuracy_score(y_true, lr_preds),
                             "f1": f1_score(y_true, lr_preds, average="macro", zero_division=0)},
    "Custom CNN":          {"preds": cnn_preds, "acc": accuracy_score(y_true, cnn_preds),
                             "f1": f1_score(y_true, cnn_preds, average="macro", zero_division=0)},
    "ResNet-18":           {"preds": rn_preds,  "acc": accuracy_score(y_true, rn_preds),
                             "f1": f1_score(y_true, rn_preds, average="macro", zero_division=0)},
}

best_name = max(model_results, key=lambda k: model_results[k]["f1"])
print(f"Best model by macro-F1: {best_name}\n")

# ---------------------------------------------------------------------------
# Figure 1 — Training curves (CNN + ResNet)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training Curves", fontsize=13, fontweight="bold")

for col, (hist, name) in enumerate([(cnn_res, "CNN"), (rn_res, "ResNet-18")]):
    ep = range(1, len(hist["train_loss"]) + 1)
    axes[0][col].plot(ep, hist["train_loss"], label="Train loss")
    axes[0][col].plot(ep, hist["val_loss"],   label="Val loss")
    axes[0][col].set_title(f"{name} — Loss")
    axes[0][col].set_xlabel("Epoch"); axes[0][col].legend()
    axes[0][col].spines[["top","right"]].set_visible(False)

    axes[1][col].plot(ep, hist["val_acc"], label="Val acc",  color="steelblue")
    axes[1][col].plot(ep, hist["val_f1"],  label="Val macro-F1", color="darkorange")
    axes[1][col].set_title(f"{name} — Val Metrics")
    axes[1][col].set_xlabel("Epoch"); axes[1][col].legend()
    axes[1][col].axhline(0.60, color="red", linestyle="--", linewidth=0.8, label="F1=0.60 target")
    axes[1][col].spines[["top","right"]].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / "training_curves.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'training_curves.png'}")

# ---------------------------------------------------------------------------
# Figure 2 — Confusion matrices (3 side by side)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")

short_labels = ["Low", "LM", "UM", "High"]
for ax, (mname, mdata) in zip(axes, model_results.items()):
    cm = confusion_matrix(y_true, mdata["preds"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        ax=ax, cbar=False, linewidths=0.5,
    )
    ax.set_title(f"{mname}\nacc={mdata['acc']:.3f}  macro-F1={mdata['f1']:.3f}", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

fig.tight_layout()
fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'confusion_matrices.png'}")

# ---------------------------------------------------------------------------
# Figure 3 — Model comparison bar chart
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle("Model Comparison on Test Set", fontsize=12, fontweight="bold")
model_names = list(model_results.keys())
colors = ["#4e79a7", "#f28e2b", "#59a14f"]

for ax, metric in zip(axes, ["acc", "f1"]):
    vals = [model_results[m][metric] for m in model_names]
    bars = ax.bar(model_names, vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0.60, color="red", linestyle="--", linewidth=1, label="F1=0.60 target")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    title = "Accuracy" if metric == "acc" else "Macro-F1"
    ax.set_title(title); ax.set_ylim(0, 1.05)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    if metric == "f1":
        ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "model_comparison.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'model_comparison.png'}")

# ---------------------------------------------------------------------------
# Figure 4 — Feature importance (LogReg coefficients)
# ---------------------------------------------------------------------------

coef_df = pd.DataFrame(
    logreg.coef_,
    index=TIER_ORDER,
    columns=FEATURE_COLS,
)
fig, ax = plt.subplots(figsize=(11, 3.5))
sns.heatmap(
    coef_df, annot=True, fmt=".2f",
    cmap=sns.diverging_palette(10, 240, as_cmap=True),
    center=0, linewidths=0.4, ax=ax,
    cbar_kws={"label": "Coefficient"},
)
ax.set_title("Logistic Regression Coefficients per Income Tier", fontsize=11)
ax.set_xticklabels([c.replace("_","\n") for c in FEATURE_COLS], fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'feature_importance.png'}")

# ---------------------------------------------------------------------------
# Grad-CAM helper
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, img_tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(img_tensor)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        score = out[0, class_idx]
        score.backward()
        weights  = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam      = (weights * self.activations).sum(dim=1).squeeze()
        cam      = torch.relu(cam)
        cam      = cam - cam.min()
        denom    = cam.max()
        if denom > 1e-8:
            cam = cam / denom
        return cam.cpu().numpy(), class_idx


def gradcam_figure(model, target_layer, test_loader, title_prefix, out_path, n_samples=8):
    gcam   = GradCAM(model, target_layer)
    items  = []
    model.train()   # keep BN in training mode so gradients flow properly
    for imgs, labels, iso3s in test_loader:
        for i in range(len(imgs)):
            img   = imgs[i:i+1].to(DEVICE).requires_grad_(True)
            cam, pred = gcam(img)
            items.append({
                "iso3": iso3s[i],
                "true": labels[i].item(),
                "pred": pred,
                "img":  imgs[i, 0].numpy(),
                "cam":  cam,
            })
            if len(items) >= n_samples:
                break
        if len(items) >= n_samples:
            break

    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 4, n_rows * 3))
    fig.suptitle(f"{title_prefix} — Grad-CAM (test set)", fontsize=11, fontweight="bold")

    for idx, item in enumerate(items):
        row, col_base = divmod(idx, n_cols)
        col_base *= 2

        ax_img = axes[row][col_base]     if n_rows > 1 else axes[col_base]
        ax_cam = axes[row][col_base + 1] if n_rows > 1 else axes[col_base + 1]

        true_label = TIER_ORDER[item["true"]]
        pred_label = TIER_ORDER[item["pred"]]
        correct    = item["true"] == item["pred"]

        ax_img.imshow(item["img"], cmap="inferno", vmin=0, vmax=1)
        ax_img.set_title(f"{item['iso3']}\nTrue: {true_label[:4]}",
                         fontsize=7, color="green" if correct else "red")
        ax_img.axis("off")

        ax_cam.imshow(item["img"], cmap="inferno", vmin=0, vmax=1)
        ax_cam.imshow(item["cam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax_cam.set_title(f"Pred: {pred_label[:4]}", fontsize=7,
                         color="green" if correct else "red")
        ax_cam.axis("off")

    # hide unused axes
    total_axes = n_rows * n_cols * 2
    for idx in range(len(items) * 2, total_axes):
        r, c = divmod(idx, n_cols * 2)
        ax = axes[r][c] if n_rows > 1 else axes[c]
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# Grad-CAM for CNN (last conv layer = features[-1] which is MaxPool; use features[-2]=ReLU before it)
# We target the last Conv block's ReLU output: index 10 = final ReLU
cnn_target = cnn_model.features[10]   # ReLU after 3rd conv block
te_cnn_full = DataLoader(NTLDataset(test_iso, three_channel=False), batch_size=1)
gradcam_figure(cnn_model, cnn_target, te_cnn_full,
               "Custom CNN", FIG_DIR / "gradcam_cnn.png")

# Grad-CAM for ResNet (last residual layer)
rn_target = resnet.layer4[-1]
te_rn_full = DataLoader(NTLDataset(test_iso, three_channel=True), batch_size=1)
gradcam_figure(resnet, rn_target, te_rn_full,
               "ResNet-18", FIG_DIR / "gradcam_resnet.png")

# ---------------------------------------------------------------------------
# Console interpretation
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("EVALUATION SUMMARY")
print("=" * 65)

TARGET_F1 = 0.60
for mname, mdata in model_results.items():
    flag = "PASS" if mdata["f1"] >= TARGET_F1 else "BELOW TARGET"
    print(f"  {mname:<22}  acc={mdata['acc']:.4f}  macro-F1={mdata['f1']:.4f}  [{flag}]")

print(f"\n  Target macro-F1 >= {TARGET_F1}")

# Confusion-based analysis: find hardest tier pairs for best model
best_preds = model_results[best_name]["preds"]
cm = confusion_matrix(y_true, best_preds)
off_diag = [(cm[i, j], TIER_ORDER[i], TIER_ORDER[j])
            for i in range(4) for j in range(4) if i != j and cm[i, j] > 0]
off_diag.sort(reverse=True)

print(f"\n  Hardest misclassifications ({best_name}):")
for count, true_t, pred_t in off_diag[:5]:
    print(f"    True={true_t:<16} → Pred={pred_t:<16} : {count} countries")

# LM vs UM boundary check
lm_idx, um_idx = TIER2IDX["Lower-Middle"], TIER2IDX["Upper-Middle"]
lm_um_err = cm[lm_idx, um_idx] + cm[um_idx, lm_idx]
print(f"\n  Lower-Middle ↔ Upper-Middle confusions: {lm_um_err}")
if lm_um_err >= 3:
    print("  → Boundary largely unresolvable from nighttime light alone.")

# Outlier analysis
print(f"\n  Notable test-set country predictions ({best_name}):")
outliers_of_interest = {"SYR", "PSE", "PRK", "LBY", "VEN", "NOR", "LUX"}
for iso3, true_idx, pred_idx in zip(test_iso, y_true, best_preds):
    if iso3 in outliers_of_interest or true_idx != pred_idx:
        true_t = TIER_ORDER[true_idx]
        pred_t = TIER_ORDER[pred_idx]
        flag   = "" if true_idx == pred_idx else "  ← MISCLASSIFIED"
        print(f"    {iso3:<6} True={true_t:<16} Pred={pred_t:<16}{flag}")

# LogReg vs CNN signal
lr_f1  = model_results["Logistic Regression"]["f1"]
cnn_f1 = model_results["Custom CNN"]["f1"]
delta  = cnn_f1 - lr_f1
print(f"\n  CNN vs LogReg macro-F1 delta: {delta:+.4f}")
if delta > 0.03:
    print("  → Raw image patterns add meaningful signal beyond hand-crafted features.")
elif delta < -0.03:
    print("  → Hand-crafted features outperform raw CNN — spatial metrics carry most signal.")
else:
    print("  → Marginal difference; hand-crafted features capture most available signal.")

print("=" * 65)
print("All figures saved to figures/")
