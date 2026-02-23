import sys
print(sys.executable)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import os
import pickle
import numpy as np
import random
import time


seed = 69  # You can choose any integer seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

'''# If using CUDA, set the seed for GPU as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
'''

def train_and_evaluate(model, trainloader, testloader, optimizer, loss_fn, num_epochs, corr_p, rep = None, sound=10, loss_type = None, clothing = False, method_name=None):
    seed = 42  # You can choose any integer seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    initial_lr = optimizer.param_groups[0]['lr']

    # Initialize a list to store epoch data
    results = []

    #Only when debbuging
    #torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct_train = 0

        for inputs, vl, targets in trainloader:

        #    if i == 0:
        #       for name, param in model.named_parameters():
        #          print(name, param)

            #if loss_type == 'Supervised':
            #    train_targets = torch.max(targets, dim=1)[1]
            inputs, vl, targets = inputs.to(device), vl.to(device), targets.to(device)
            

            optimizer.zero_grad()
            outputs = model(inputs)
            #if loss_type == 'Supervised':
            #    # For cross-entropy loss, targets should be class indices
            #    loss = loss_fn(outputs, train_targets)
            #else:
            loss = loss_fn.forward(outputs, vl) 
            loss.backward()
            optimizer.step()

            # Update batch's loss and accuracy
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)
            correct_train += torch.sum(preds == true)

        train_acc = correct_train.double() / len(trainloader.dataset)
        train_loss = running_loss / len(trainloader.dataset)

        # Evaluate the model on the test set
        model.eval()
        correct_test = 0
        with torch.no_grad():
            for inputs, targets in testloader: 
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                _, preds = torch.max(outputs, dim=1)
                _, true = torch.max(targets, dim=1)
                correct_test += torch.sum(preds == true)

        test_acc = correct_test.double() / len(testloader.dataset)

        # Calculate detached loss 
        detached_train_loss = 0.0
        detached_test_loss = 0.0
        with torch.no_grad():
            det_loss_fn = torch.nn.CrossEntropyLoss()  
            for inputs, _, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                detached_train_loss += det_loss_fn(outputs, targets).item()
            detached_train_loss /= len(trainloader.dataset)

            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                detached_test_loss += det_loss_fn(outputs, targets).item()
            detached_test_loss /= len(testloader.dataset)

        # Get the actual learning rate from the optimizer
        actual_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        # Store results for this epoch
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),
            'test_acc': test_acc.item(),
            'train_detached_loss': detached_train_loss,
            'test_detached_loss': detached_test_loss,
            'optimizer': type(optimizer).__name__,
            'loss_fn': loss_type,
            'method': method_name if method_name is not None else type(loss_fn).__name__,
            'repetition': rep,
            'initial_lr': initial_lr,
            'actual_lr': actual_lr,
            'corr_p': corr_p,
            'epoch_time': epoch_time,
        }
        results.append(epoch_data)

        if epoch % sound == sound - 1:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
                  f'Train Detached Loss: {detached_train_loss:.4f}, Test Detached Loss: {detached_test_loss:.4f}, '
                  f'Learning Rate: {actual_lr:.6f}, Epoch Time: {epoch_time:.2f} seconds')

    # Convert results to DataFrame at the end
    results_df = pd.DataFrame(results)

    return model, results_df


# ---------------- Plotting utilities ----------------
# These helpers create a single figure that compares:
#   - Accuracy curves: Marginal Chain (MC) vs Forward Proper Loss (FWD) on the same axes
#   - Loss curves: one subplot per method
#
# Expected inputs: the DataFrames returned by train_and_evaluate() for each method.

from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _pretty_loss_name(loss_code: str) -> str:
    if loss_code is None:
        return "loss"
    if loss_code.startswith("ps_"):
        beta = loss_code.split("_", 1)[1]
        return f"PS(p={beta})"
    if loss_code.startswith("tsallis_"):
        a = loss_code.split("_", 1)[1]
        return f"Tsallis(α={a})"
    mapping = {
        "cross_entropy": "Cross-Entropy",
        "brier": "Brier",
        "squared": "Squared",
        "mse": "MSE",
        "spherical": "Spherical",
    }
    return mapping.get(loss_code, loss_code)


def _aggregate_by_epoch(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (epochs, mean, std). Works with single-run or multi-rep DataFrames."""
    if df is None or len(df) == 0:
        raise ValueError("Empty DataFrame passed to _aggregate_by_epoch")

    # Ensure sorted
    df = df.sort_values(["epoch"] + (["repetition"] if "repetition" in df.columns else []))

    g = df.groupby("epoch")[col]
    epochs = g.mean().index.to_numpy()
    mean = g.mean().to_numpy()
    std = g.std(ddof=0).fillna(0.0).to_numpy()  # std=0 for single run
    return epochs, mean, std


def plot_mc_vs_forward(
    mc_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    base_loss_code: Optional[str] = None,
    save_path: Optional[str] = None,
    notes: Optional[Dict[str, object]] = None,
    show: bool = True,
    show_notes: bool = False,
):
    """
    Make a figure like:

      [ Accuracy (MC vs FWD) ]
      [ MC train loss ] [ FWD train loss ]  (optional notes box)

    Parameters
    ----------
    mc_df, fwd_df:
        DataFrames returned by train_and_evaluate() for each method.
    base_loss_code:
        e.g. "cross_entropy", "brier", "ps_2". If None, will try to infer from df['loss_fn'].
    save_path:
        If provided, save figure to this path (png/pdf supported by matplotlib).
    notes:
        Dict of key/value pairs to render in a note box at bottom-right.
    show:
        Whether to call plt.show().
    show_notes:
        If True, render the note box (only if `notes` is provided).
    """
    if base_loss_code is None:
        for df in (mc_df, fwd_df):
            if df is not None and "loss_fn" in df.columns and df["loss_fn"].notna().any():
                base_loss_code = df["loss_fn"].dropna().iloc[0]
                break

    title = f"Fig. 1. {_pretty_loss_name(str(base_loss_code))} — MC vs FWD"

    # fig = plt.figure(figsize=(10.5, 7.0), dpi=150)
    # gs = GridSpec(nrows=2, ncols=2, height_ratios=[1.25, 1.0], hspace=0.35, wspace=0.25)

    # ax_acc = fig.add_subplot(gs[0, :])
    # ax_mc = fig.add_subplot(gs[1, 0])
    # ax_fwd = fig.add_subplot(gs[1, 1])

fig = plt.figure(figsize=(11.5, 6.0), dpi=150)
gs = GridSpec(
    nrows=2,
    ncols=2,
    width_ratios=[1.2, 1.0],
    hspace=0.30,
    wspace=0.30,
)

# 左边：Accuracy（跨两行）
ax_acc = fig.add_subplot(gs[:, 0])
ax_acc.set_title("Accuracy comparison", fontsize=12, fontweight="bold")

# 右边：MC / FWD train loss
ax_mc  = fig.add_subplot(gs[0, 1])
ax_fwd = fig.add_subplot(gs[1, 1])



    # -------- Accuracy (use test_acc by default; also overlay train_acc as dashed if present) --------
    for df, label, color in [
        (mc_df, "MC", "orange"),
        (fwd_df, "FWD", "tab:blue"),
    ]:
        if df is None or len(df) == 0:
            continue

        if "test_acc" in df.columns:
            ep, mean, std = _aggregate_by_epoch(df, "test_acc")
            ax_acc.plot(ep, mean, label=f"{label} test", color=color, linewidth=2)
            if np.any(std > 0):
                ax_acc.fill_between(ep, mean - std, mean + std, color=color, alpha=0.18)

        if "train_acc" in df.columns:
            ep, mean, std = _aggregate_by_epoch(df, "train_acc")
            ax_acc.plot(ep, mean, label=f"{label} train", color=color, linestyle="--", linewidth=1.6)

    ax_acc.set_title(title)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(loc="lower right", frameon=True)

    # -------- Loss curves (train_loss only) --------
    def _plot_losses(ax, df: pd.DataFrame, method_label: str, color: str):
        if df is None or len(df) == 0:
            ax.set_title(f"{method_label} losses (no data)")
            ax.axis("off")
            return

        # Main objective curve
        if "train_loss" in df.columns:
            ep, mean, std = _aggregate_by_epoch(df, "train_loss")
            ax.plot(ep, mean, color=color, linewidth=2)

        ax.set_title(f"{method_label} train loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

    _plot_losses(ax_mc, mc_df, "MC", "orange")
    _plot_losses(ax_fwd, fwd_df, "FWD", "tab:blue")

    # -------- Notes box (bottom-right) --------
    if show_notes and notes:
        # Create a small, axis-less box in figure coords
        ax_note = fig.add_axes([0.70, 0.03, 0.28, 0.26])
        ax_note.axis("off")
        note_lines = [f"{k}: {v}" for k, v in notes.items()]
        note_text = "\n".join(note_lines)
        ax_note.text(
            0.0, 1.0, note_text,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="0.6"),
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig
