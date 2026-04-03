"""Plot training curves from CSV logs produced by the fine-tuning pipeline.

Reads ``train_log_steps.csv`` and ``train_log_epochs.csv`` from a run
directory and generates publication-quality PNG charts.

Usage (from repo root)::

    python3.12 -m scripts.plot_training --run-dir outputs/pipeline_run

Produces:
    outputs/pipeline_run/plots/
        loss_vs_step.png        — step-level training loss curve
        loss_vs_epoch.png       — epoch-level train vs val loss
        lr_vs_step.png          — learning rate schedule
        epoch_duration.png      — wall-clock time per epoch
        training_dashboard.png  — all-in-one summary panel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib is required.  pip install matplotlib")
    sys.exit(1)

import csv
import json
from typing import Any


# ── Consistent style ────────────────────────────────────────────────────────

_COLOR_TRAIN = "#2563eb"
_COLOR_VAL = "#dc2626"
_COLOR_LR = "#7c3aed"
_COLOR_DURATION = "#059669"
_COLOR_BEST = "#f59e0b"


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


# ── CSV readers ─────────────────────────────────────────────────────────────

def _read_step_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step": int(row["step"]),
                "epoch": int(row["epoch"]),
                "loss": float(row["loss"]),
                "lr": float(row["lr"]),
                "elapsed_sec": float(row["elapsed_sec"]),
            })
    return rows


def _read_epoch_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("val_loss", "")
            rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(val) if val not in ("", "None") else None,
                "best_loss": float(row["best_loss"]),
                "steps_in_epoch": int(row["steps_in_epoch"]),
                "lr": float(row["lr"]),
                "epoch_duration_sec": float(row["epoch_duration_sec"]),
                "total_elapsed_sec": float(row["total_elapsed_sec"]),
            })
    return rows


# ── Individual chart functions ──────────────────────────────────────────────

def plot_loss_vs_step(step_data: list[dict], out: Path) -> None:
    """Training loss at each logging step, with a smoothed trend line."""
    if not step_data:
        return

    steps = [d["step"] for d in step_data]
    losses = [d["loss"] for d in step_data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color=_COLOR_TRAIN, alpha=0.35, linewidth=0.8, label="raw")

    # exponential moving average for a smooth trend
    alpha = 0.05
    ema = [losses[0]]
    for v in losses[1:]:
        ema.append(ema[-1] * (1 - alpha) + v * alpha)
    ax.plot(steps, ema, color=_COLOR_TRAIN, linewidth=2, label="smoothed (EMA)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training Loss vs. Step")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.savefig(str(out))
    plt.close(fig)


def plot_loss_vs_epoch(epoch_data: list[dict], out: Path) -> None:
    """Train loss and validation loss per epoch."""
    if not epoch_data:
        return

    epochs = [d["epoch"] for d in epoch_data]
    train = [d["train_loss"] for d in epoch_data]
    best = [d["best_loss"] for d in epoch_data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train, color=_COLOR_TRAIN, marker="o", markersize=3, linewidth=1.5, label="Train Loss")

    has_val = any(d["val_loss"] is not None for d in epoch_data)
    if has_val:
        val_epochs = [d["epoch"] for d in epoch_data if d["val_loss"] is not None]
        val_losses = [d["val_loss"] for d in epoch_data if d["val_loss"] is not None]
        ax.plot(val_epochs, val_losses, color=_COLOR_VAL, marker="s", markersize=3, linewidth=1.5, label="Val Loss")

    ax.plot(epochs, best, color=_COLOR_BEST, linestyle=":", linewidth=1, alpha=0.7, label="Best Train Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Loss vs. Epoch")
    ax.legend()

    fig.savefig(str(out))
    plt.close(fig)


def plot_lr_vs_step(step_data: list[dict], out: Path) -> None:
    """Learning rate over training steps."""
    if not step_data:
        return

    steps = [d["step"] for d in step_data]
    lrs = [d["lr"] for d in step_data]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(steps, lrs, color=_COLOR_LR, linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.savefig(str(out))
    plt.close(fig)


def plot_epoch_duration(epoch_data: list[dict], out: Path) -> None:
    """Wall-clock seconds per epoch as a bar chart."""
    if not epoch_data:
        return

    epochs = [d["epoch"] for d in epoch_data]
    durations = [d["epoch_duration_sec"] for d in epoch_data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(epochs, durations, color=_COLOR_DURATION, alpha=0.75, width=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("Epoch Duration")

    fig.savefig(str(out))
    plt.close(fig)


def plot_dashboard(
    step_data: list[dict],
    epoch_data: list[dict],
    metrics: dict[str, Any] | None,
    out: Path,
) -> None:
    """All-in-one 2x2 summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # ── top-left: step loss ──
    ax = axes[0][0]
    if step_data:
        steps = [d["step"] for d in step_data]
        losses = [d["loss"] for d in step_data]
        ax.plot(steps, losses, color=_COLOR_TRAIN, alpha=0.3, linewidth=0.7)
        alpha = 0.05
        ema = [losses[0]]
        for v in losses[1:]:
            ema.append(ema[-1] * (1 - alpha) + v * alpha)
        ax.plot(steps, ema, color=_COLOR_TRAIN, linewidth=2)
    ax.set_title("Loss vs. Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    # ── top-right: epoch train/val loss ──
    ax = axes[0][1]
    if epoch_data:
        epochs = [d["epoch"] for d in epoch_data]
        train = [d["train_loss"] for d in epoch_data]
        ax.plot(epochs, train, color=_COLOR_TRAIN, marker="o", markersize=2, linewidth=1.5, label="Train")
        has_val = any(d["val_loss"] is not None for d in epoch_data)
        if has_val:
            ve = [d["epoch"] for d in epoch_data if d["val_loss"] is not None]
            vl = [d["val_loss"] for d in epoch_data if d["val_loss"] is not None]
            ax.plot(ve, vl, color=_COLOR_VAL, marker="s", markersize=2, linewidth=1.5, label="Val")
        ax.legend(loc="upper right")
    ax.set_title("Loss vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # ── bottom-left: learning rate ──
    ax = axes[1][0]
    if step_data:
        steps = [d["step"] for d in step_data]
        lrs = [d["lr"] for d in step_data]
        ax.plot(steps, lrs, color=_COLOR_LR, linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))

    # ── bottom-right: summary text ──
    ax = axes[1][1]
    ax.axis("off")
    if metrics:
        lines = [
            f"Total Steps:        {metrics.get('total_steps', '?'):>10,}",
            f"Total Epochs:       {metrics.get('total_epochs', '?'):>10}",
            f"Train Samples:      {metrics.get('train_sample_count', '?'):>10}",
            f"Val Samples:        {metrics.get('val_sample_count', '?'):>10}",
            f"Best Train Loss:    {metrics.get('best_train_loss', '?'):>10}",
            f"Final Val Loss:     {str(metrics.get('final_val_loss', 'N/A')):>10}",
            f"Duration:           {_fmt_duration(metrics.get('total_duration_sec')):>10}",
        ]
        ax.text(
            0.1, 0.5, "\n".join(lines),
            transform=ax.transAxes, fontsize=12, fontfamily="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0", edgecolor="#cccccc"),
        )
    ax.set_title("Summary")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out))
    plt.close(fig)


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ── Main entry ──────────────────────────────────────────────────────────────

def generate_plots(run_dir: str | Path) -> Path:
    """Read training logs from ``run_dir`` and write PNGs to ``run_dir/plots/``.

    Returns the plots directory path.
    """
    _apply_style()

    run = Path(run_dir)
    plots_dir = run / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    step_csv = run / "train_log_steps.csv"
    epoch_csv = run / "train_log_epochs.csv"
    metrics_json = run / "train_metrics.json"

    step_data: list[dict] = []
    epoch_data: list[dict] = []
    metrics: dict[str, Any] | None = None

    if step_csv.exists():
        step_data = _read_step_csv(step_csv)
    if epoch_csv.exists():
        epoch_data = _read_epoch_csv(epoch_csv)
    if metrics_json.exists():
        with metrics_json.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    if not step_data and not epoch_data:
        print(f"No training logs found in {run}. Nothing to plot.")
        return plots_dir

    if step_data:
        plot_loss_vs_step(step_data, plots_dir / "loss_vs_step.png")
        plot_lr_vs_step(step_data, plots_dir / "lr_vs_step.png")
        print(f"  loss_vs_step.png   ({len(step_data)} data points)")
        print(f"  lr_vs_step.png")

    if epoch_data:
        plot_loss_vs_epoch(epoch_data, plots_dir / "loss_vs_epoch.png")
        plot_epoch_duration(epoch_data, plots_dir / "epoch_duration.png")
        print(f"  loss_vs_epoch.png  ({len(epoch_data)} epochs)")
        print(f"  epoch_duration.png")

    plot_dashboard(step_data, epoch_data, metrics, plots_dir / "training_dashboard.png")
    print(f"  training_dashboard.png")

    print(f"\nAll plots saved to {plots_dir}")
    return plots_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from CSV logs.")
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help="Run output directory containing train_log_steps.csv / train_log_epochs.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_plots(args.run_dir)


if __name__ == "__main__":
    main()
