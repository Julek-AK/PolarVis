# Builtins
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import time

# External
from numpy.typing import NDArray
import numpy as np
import torch

# Internal
from tests.processing_tests.test_case_generator import TestCase, TestCaseGenerator


@dataclass
class TestResult:
    image: NDArray
    ground_truth: Optional[NDArray]
    reconstruction: Optional[NDArray]

    metrics: dict
    metadata: dict
    success: bool
    error: Optional[Exception] = None


class TestRunner:
    def __init__(self, solver_callable, solver_kwargs=None) -> None:
        """
        solver_callable takes in an image and returns its reconstruction data.
        Make sure it follows the function signature of:
        NDArray of shape (img_size, img_size) -> NDArray of shape (img_size/2, img_size/2, 3)

        solver_kwargs: {"device": 'cpu', "eps": 1e-12}
        """
        self.solver = solver_callable
        self.solver_kwargs = solver_kwargs or {}

    def run_test_case(self, case: TestCase) -> TestResult:
        # Generate the reconstruction
        try:
            # Measure runtime
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            reconstruction = self.solver(case.image, **self.solver_kwargs)  # Call the metapixel resolving function
            end_evt.record()

            torch.cuda.synchronize()
            runtime_ms = start_evt.elapsed_time(end_evt)

        except Exception as e:
            return TestResult(
                image=case.image,
                ground_truth=case.ground_truth,
                reconstruction=None,
                metrics={},
                metadata=case.metadata,
                success=False,
                error=e,
            )

        # Compute metrics if ground truth is available
        if case.ground_truth is not None and reconstruction is not None:
            metrics = self.compute_metrics2(case.ground_truth, reconstruction)
        else:
            metrics = {}

        metrics["runtime_sec"] = runtime_ms / 1000

        return TestResult(
            image=case.image,
            ground_truth=case.ground_truth,
            reconstruction=reconstruction,
            metrics=metrics,
            metadata=case.metadata,
            success=True,
        )

    def compute_metrics(self, gt, rec):
        metrics = {}
        metrics["rmse"] = float(np.sqrt(np.mean((gt - rec)**2)))

        # Compute individual channel errors
        metrics["rmse_pol"] = float(np.sqrt(np.mean((gt[..., 0] - rec[..., 0])**2)))
        metrics["rmse_unpol"] = float(np.sqrt(np.mean((gt[..., 1] - rec[..., 1])**2)))

        # Angle error
        dtheta = np.angle(np.exp(1j * (gt[..., 2] - rec[..., 2])))
        metrics["mae_theta"] = float(np.mean(np.abs(dtheta)))

        # Statistics for debugging
        metrics["max_error"] = float(np.max(np.abs(gt - rec)))
        metrics["min_error"] = float(np.min(np.abs(gt - rec)))
        return metrics
    
    def compute_metrics2(self, gt, rec):
        metrics = {}

        gt_arr = np.asarray(gt)
        rec_arr = np.asarray(rec)

        # Difference for intensity channels
        diff_ch0 = gt_arr[..., 0] - rec_arr[..., 0]
        diff_ch1 = gt_arr[..., 1] - rec_arr[..., 1]

        # Circular difference for the theta channel
        dtheta = 0.5 * np.angle(np.exp(1j * 2.0 * (gt_arr[..., 2] - rec_arr[..., 2])))

        # Compute the metrics
        stacked_diff = np.stack([diff_ch0, diff_ch1, dtheta], axis=-1)
        metrics["rmse"] = float(np.sqrt(np.mean(stacked_diff**2)))

        metrics["rmse_pol"] = float(np.sqrt(np.mean(diff_ch0**2)))
        metrics["rmse_unpol"] = float(np.sqrt(np.mean(diff_ch1**2)))
        metrics["rmse_theta"] = float(np.sqrt(np.mean(dtheta**2)))
        metrics["mae_theta"] = float(np.mean(np.abs(dtheta)))
        
        metrics["max_error_pol"] = float(np.max(np.abs(diff_ch0)))
        metrics["min_error_pol"] = float(np.min(np.abs(diff_ch0)))

        metrics["max_error_unpol"] = float(np.max(np.abs(diff_ch1)))
        metrics["min_error_unpol"] = float(np.min(np.abs(diff_ch1)))

        metrics["max_error_theta"] = float(np.max(np.abs(dtheta)))
        metrics["min_error_theta"] = float(np.min(np.abs(dtheta)))

        return metrics


class TestBatchRunner:
    def __init__(self, runner: TestRunner) -> None:
        self.runner = runner

    def run_batch(self, generator: TestCaseGenerator, n: int = 1, seeds=None) -> List[TestResult]:
        results = []
        if seeds is None:
            seeds = [None] * n

        for i in range(n):
            case = generator.generate_case(seed=seeds[i])
            result = self.runner.run_test_case(case)
            results.append(result)

        return results