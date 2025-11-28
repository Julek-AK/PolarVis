# Builtins
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import time
import os
from PIL.Image import Image

# External
from numpy.typing import NDArray
import numpy as np
import torch
from matplotlib import pyplot as plt

# Internal
from tests.processing_tests.test_case_generator import TestCase, TestCaseGenerator
from tests.processing_tests.test_runner import TestResult, TestRunner, TestBatchRunner
from paths import TEST_OUT_DIR
from processing.visualisation import polarimetric_colormap



class TestAnalyzer:
    def __init__(self, results: list[TestResult]) -> None:
        self.results = results

    def metric_array(self, metric_name):
        arr = []
        for r in self.results:
            if metric_name in r.metrics:
                arr.append(r.metrics[metric_name])
        return np.array(arr)

    def summary(self):
        out = {"success_rate": np.mean([r.success for r in self.results])}

        # Common metrics
        for key in ["rmse", "rmse_pol", "rmse_unpol", "mae_theta", "runtime_sec"]:
            arr = self.metric_array(key)
            if len(arr):
                out[key] = {
                    "mean": arr.mean(),
                    "median": np.median(arr),
                    "std": arr.std(),
                    "min": arr.min(),
                    "max": arr.max(),
                }

        return out

    def worst_cases(self, metric="rmse", top_k=5):
        scored = [(r.metrics.get(metric, np.inf), r) for r in self.results]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]
    

def visualize_test_result(result: TestResult, test_id: str):
    """
    Saves a visualization PNG summarizing the test result:
      - metadata + metrics at top
      - input image
      - reconstruction (polarimetric colormap)
      - ground-truth (polarimetric colormap) if available
    """
    # Output directory
    os.makedirs(TEST_OUT_DIR, exist_ok=True)
    save_path = os.path.join(TEST_OUT_DIR, f"{test_id}.png")

    # Prepare and sanitize inputs
    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    img = result.image
    rec = result.reconstruction
    gt = result.ground_truth   # may be None
    metrics = result.metrics or {}
    metadata = result.metadata or {}

    img = to_numpy(img)
    rec = to_numpy(rec)
    gt  = to_numpy(gt)

    # Get the colormaps
    rec_cmap = polarimetric_colormap(rec) if rec is not None else None
    gt_cmap  = polarimetric_colormap(gt)  if gt is not None else None

    # Assemble the figure
    n_cols = 3 if gt is not None else 2
    fig = plt.figure(figsize=(5 * n_cols, 12))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.2, 6])

    # ---- HEADER (metadata + metrics) ----
    ax_header = fig.add_subplot(grid[0, 0])
    ax_header.axis("off")

    text = f"Test Result: {test_id}\n\nMetadata:"
    for k, v in metadata.items():
        text += f"\n  • {k}: {v}"

    text += "\n\nMetrics:"
    for k, v in metrics.items():
        text += f"\n  • {k}: {v}"

    if not result.success:
        text += f"\n\n FAILED: {result.error}"

    ax_header.text(
        0.02, 0.98,
        text,
        va="top", ha="left",
        fontsize=11,
        family="monospace"
    )

    # ---- IMAGE PANEL (columns) ----
    img_grid = grid[1, 0].subgridspec(1, n_cols, wspace=0.15)

    # Column 1: Input
    ax1 = fig.add_subplot(img_grid[0, 0])
    ax1.imshow(img.squeeze(), cmap="gray" if img.ndim == 2 else None)
    ax1.set_title("Input Image")
    ax1.axis("off")

    # Column 2: Reconstruction
    ax2 = fig.add_subplot(img_grid[0, 1])
    if rec is not None:
        ax2.imshow(rec_cmap)
        ax2.set_title("Reconstruction (polarimetric colormap)")
    else:
        ax2.text(0.5, 0.5, "No Reconstruction", ha="center", va="center")
        ax2.set_title("Reconstruction (missing)")
    ax2.axis("off")

    # Column 3: Ground Truth (optional)
    if gt is not None:
        ax3 = fig.add_subplot(img_grid[0, 2])
        ax3.imshow(gt_cmap)
        ax3.set_title("Ground Truth (polarimetric colormap)")
        ax3.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)