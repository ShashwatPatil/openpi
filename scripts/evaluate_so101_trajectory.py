import dataclasses
import logging
import pathlib
import time
from typing import List, Literal
import warnings

import jax
import jax.numpy as jnp
from lerobot.datasets import lerobot_dataset
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
import openpi.transforms as _transforms

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclasses.dataclass
class EvalConfig:
    """Configuration for evaluating SO101 policy."""

    config_name: str = "pi0_so101_lora"
    """Config name to use for evaluation."""

    checkpoint_path: str = "./pi0_lora_so101_checkpoints/so101_experiment/19999"
    """Path to the trained model checkpoint."""

    dataset_repo: str = "SGPatil/so101_table_cleanup_2"
    """HuggingFace dataset repository."""

    num_trajectories: int = 5
    """Number of trajectories to evaluate."""

    max_steps_per_traj: int = 50
    """Maximum steps to evaluate per trajectory."""

    action_horizon: int = 50
    """Action horizon for the model."""

    plot: bool = True
    """Whether to plot action trajectories."""

    save_plots: bool = True
    """Whether to save plots to disk."""

    output_dir: str = "./eval_results"
    """Directory to save evaluation results."""

    skip_norm_stats: bool = False
    """Whether to skip normalization stats."""

    prediction_visualization: bool = True
    """Whether to show prediction points on the trajectory plot."""

    show_prediction_horizon: bool = True
    """Whether to show the full action horizon prediction."""


def calc_action_mse(pred_actions: np.ndarray, gt_actions: np.ndarray, action_dim: int = 6) -> dict:
    """Calculate MSE between predicted and ground truth actions."""
    # Only compare the first action_dim dimensions (SO101 specific)
    pred_actions = pred_actions[:, :action_dim]
    gt_actions = gt_actions[:, :action_dim]

    # Calculate MSE for each action dimension
    mse_per_dim = np.mean((pred_actions - gt_actions) ** 2, axis=0)
    overall_mse = np.mean(mse_per_dim)

    return {
        "overall_mse": overall_mse,
        "mse_per_dim": mse_per_dim,
        "mae": np.mean(np.abs(pred_actions - gt_actions)),
        "max_error": np.max(np.abs(pred_actions - gt_actions)),
    }


def plot_action_trajectory(
    pred_actions: np.ndarray, gt_actions: np.ndarray, traj_id: int, save_path: pathlib.Path = None, action_dim: int = 6
):
    """Plot predicted vs ground truth actions for a single trajectory."""

    # Limit to SO101 action dimensions
    pred_actions = pred_actions[:, :action_dim]
    gt_actions = gt_actions[:, :action_dim]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"SO101 Action Trajectory Comparison - Trajectory {traj_id}", fontsize=16)

    action_names = ["action dim 0", "action dim 1", "action dim 2", "action dim 3", "action dim 4", "action dim 5"]

    for i in range(action_dim):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        steps = range(len(pred_actions))
        ax.plot(steps, gt_actions[:, i], "b-", label="Ground Truth", linewidth=2, alpha=0.8)
        ax.plot(steps, pred_actions[:, i], "r--", label="Predicted", linewidth=2, alpha=0.8)

        ax.set_title(f"{action_names[i]} Actions")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calculate and display MSE for this dimension
        mse = np.mean((pred_actions[:, i] - gt_actions[:, i]) ** 2)
        ax.text(
            0.02,
            0.98,
            f"MSE: {mse:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / f"trajectory_{traj_id}_actions.png", dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path / f'trajectory_{traj_id}_actions.png'}")

    if plt.get_backend() != "Agg":  # Only show if not in headless mode
        plt.show()

    plt.close()


def plot_action_trajectory_with_predictions(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    prediction_points: list,  # List of (step, prediction_array) tuples
    traj_id: int,
    save_path: pathlib.Path = None,
    action_dim: int = 6,
):
    """Plot predicted vs ground truth actions with prediction points highlighted."""

    # Limit to SO101 action dimensions
    pred_actions = pred_actions[:, :action_dim]
    gt_actions = gt_actions[:, :action_dim]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"SO101 Action Trajectory with Prediction Horizon - Trajectory {traj_id}", fontsize=16)

    action_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

    for i in range(action_dim):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        steps = range(len(gt_actions))

        # Plot ground truth trajectory
        ax.plot(steps, gt_actions[:, i], "b-", label="Ground Truth", linewidth=2, alpha=0.8)

        # Plot single-step predictions
        pred_steps = range(len(pred_actions))
        ax.plot(pred_steps, pred_actions[:, i], "r--", label="Single-step Predictions", linewidth=2, alpha=0.8)

        # Plot prediction points and horizons
        for pred_step, horizon_prediction in prediction_points:
            if pred_step < len(gt_actions):
                # Mark the prediction point on ground truth
                ax.plot(
                    pred_step,
                    gt_actions[pred_step, i],
                    "go",
                    markersize=8,
                    label="Prediction Point" if pred_step == prediction_points[0][0] else "",
                )

                # Show the predicted horizon from this point
                if horizon_prediction.shape[0] > 1:  # Multi-step prediction
                    horizon_steps = range(pred_step, min(pred_step + len(horizon_prediction), len(gt_actions)))
                    horizon_values = horizon_prediction[: len(horizon_steps), i]

                    ax.plot(
                        horizon_steps,
                        horizon_values,
                        "orange",
                        linewidth=3,
                        alpha=0.7,
                        label="Prediction Horizon" if pred_step == prediction_points[0][0] else "",
                    )

                    # Add markers for individual horizon predictions
                    ax.scatter(horizon_steps, horizon_values, c="orange", s=30, alpha=0.8, zorder=5)

        ax.set_title(f"{action_names[i]} Actions")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calculate and display MSE for this dimension
        mse = np.mean((pred_actions[:, i] - gt_actions[: len(pred_actions), i]) ** 2)
        ax.text(
            0.02,
            0.98,
            f"MSE: {mse:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / f"trajectory_{traj_id}_predictions.png", dpi=150, bbox_inches="tight")
        print(f"Saved prediction plot to {save_path / f'trajectory_{traj_id}_predictions.png'}")

    if plt.get_backend() != "Agg":
        plt.show()

    plt.close()


def evaluate_single_trajectory(
    policy, dataset, traj_id: int, max_steps: int, action_horizon: int, config: EvalConfig
) -> dict:
    """Evaluate policy on a single trajectory."""

    print(f"Evaluating trajectory {traj_id}...")

    # Get trajectory data
    episode_starts = dataset.episode_data_index["from"]
    episode_ends = dataset.episode_data_index["to"]

    traj_start_idx = int(episode_starts[traj_id])
    traj_end_idx = int(episode_ends[traj_id])
    traj_length = traj_end_idx - traj_start_idx

    actual_steps = min(max_steps, traj_length)

    predicted_actions = []
    ground_truth_actions = []
    prediction_points = []  # Store (step, full_horizon_prediction) for visualization

    # Timing measurements
    inference_times = []

    for step in range(actual_steps):
        data_idx = traj_start_idx + step

        try:
            # Get observation data
            sample = dataset[int(data_idx)]

            obs_dict = {
                "images": {
                    "front": sample["observation.images.front"],
                    "laptop": sample["observation.images.front"],  # Use front for both if laptop not available
                },
                "state": sample["observation.state"],
                "prompt": sample.get("task", "pick up the object"),
            }

            # Time the inference
            start_time = time.perf_counter()
            result = policy.infer(obs_dict)
            end_time = time.perf_counter()

            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)

            # Debug: print timing and shapes for first few steps
            if step < 3:
                print(f"Step {step} - Inference time: {inference_time_ms:.2f}ms")
                print(f"Step {step} - Result keys: {result.keys()}")
                if hasattr(result["actions"], "shape"):
                    print(f"Step {step} - Actions shape: {result['actions'].shape}")

            # Handle different action output formats
            actions = result["actions"]
            actions_np = np.asarray(actions)

            # Handle different action shapes
            if actions_np.ndim == 0:
                print(f"Warning: Got scalar action at step {step}")
                continue
            elif actions_np.ndim == 1:
                # Single action vector
                pred_action = actions_np
                full_horizon = actions_np.reshape(1, -1)  # Make it 2D for consistency
            elif actions_np.ndim == 2:
                # Action sequence - this is what we want for horizon visualization
                pred_action = actions_np[0]  # First action for execution
                full_horizon = actions_np  # Full sequence for visualization
            elif actions_np.ndim == 3:
                # Batch dimension
                pred_action = actions_np[0, 0]
                full_horizon = actions_np[0]
            else:
                print(f"Warning: Unexpected action shape at step {step}: {actions_np.shape}")
                continue

            # Store prediction point for visualization (every few steps to avoid clutter)
            if config.prediction_visualization and (step % 5 == 0 or step < 5):
                prediction_points.append((step, full_horizon))

            # Get ground truth action
            gt_action = np.asarray(sample["action"])

            # Debug: print action details for first few steps
            if step < 3:
                print(f"Step {step} - Pred action shape: {pred_action.shape}, GT action shape: {gt_action.shape}")
                print(f"Step {step} - Full horizon shape: {full_horizon.shape}")
                print(f"Step {step} - Pred action (first 6): {pred_action[:6]}")
                print(f"Step {step} - GT action: {gt_action}")

            predicted_actions.append(pred_action)
            ground_truth_actions.append(gt_action)

        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback

            traceback.print_exc()
            break

    if not predicted_actions:
        return {"error": "No valid predictions"}

    # Convert to numpy arrays
    pred_actions = np.array(predicted_actions)
    gt_actions = np.array(ground_truth_actions)

    print(f"Final arrays - Pred: {pred_actions.shape}, GT: {gt_actions.shape}")

    # Calculate metrics
    metrics = calc_action_mse(pred_actions, gt_actions, action_dim=6)

    # Add timing metrics
    if inference_times:
        mean_inference_time = np.mean(inference_times)
        metrics.update(
            {
                "mean_inference_time_ms": mean_inference_time,
                "inference_rate_hz": 1000.0 / mean_inference_time,
                "std_inference_time_ms": np.std(inference_times),
            }
        )
        print(
            f"Trajectory {traj_id} - Mean inference time: {mean_inference_time:.2f}ms ({1000.0 / mean_inference_time:.1f} Hz)"
        )

    # Plot with prediction visualization
    if config.plot:
        output_dir = pathlib.Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.save_plots:
            if config.prediction_visualization:
                plot_action_trajectory_with_predictions(
                    pred_actions, gt_actions, prediction_points, traj_id, output_dir
                )
            else:
                plot_action_trajectory(pred_actions, gt_actions, traj_id, output_dir)

    metrics.update(
        {
            "trajectory_id": traj_id,
            "steps_evaluated": len(predicted_actions),
            "trajectory_length": traj_length,
            "prediction_points": len(prediction_points),
        }
    )

    return metrics


def main(config: EvalConfig):
    """Main evaluation function with enhanced visualization and timing."""

    print(f"Starting SO101 policy evaluation...")
    print(f"Config: {config.config_name}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Dataset: {config.dataset_repo}")

    # Create output directory
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained policy
    train_config = _config.get_config(config.config_name)

    if config.checkpoint_path.startswith("gs://"):
        checkpoint_dir = download.maybe_download(config.checkpoint_path)
    else:
        checkpoint_dir = pathlib.Path(config.checkpoint_path)

    print("Loading trained policy...")
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

    # Load dataset
    print("Loading dataset...")
    dataset = lerobot_dataset.LeRobotDataset(config.dataset_repo)

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Number of episodes: {len(dataset.episode_data_index['from'])}")

    # Evaluate on multiple trajectories
    all_metrics = []
    all_inference_times = []

    num_episodes = len(dataset.episode_data_index["from"])
    actual_trajs = min(config.num_trajectories, num_episodes)

    print(f"Evaluating on {actual_trajs} trajectories...")

    for traj_id in range(actual_trajs):
        metrics = evaluate_single_trajectory(
            policy, dataset, traj_id, config.max_steps_per_traj, config.action_horizon, config
        )

        if "error" not in metrics:
            all_metrics.append(metrics)
            if "inference_rate_hz" in metrics:
                all_inference_times.append(metrics["mean_inference_time_ms"])

            print(
                f"Trajectory {traj_id}: MSE = {metrics['overall_mse']:.4f}, "
                f"MAE = {metrics['mae']:.4f}, Steps = {metrics['steps_evaluated']}"
            )
            if "inference_rate_hz" in metrics:
                print(f"  Rate: {metrics['inference_rate_hz']:.1f} Hz")
        else:
            print(f"Trajectory {traj_id}: Failed - {metrics['error']}")

    if not all_metrics:
        print("No successful evaluations!")
        return

    # Aggregate results
    overall_mse = np.mean([m["overall_mse"] for m in all_metrics])
    overall_mae = np.mean([m["mae"] for m in all_metrics])

    # Timing summary
    if all_inference_times:
        mean_time = np.mean(all_inference_times)
        mean_rate = 1000.0 / mean_time

        print(f"\n{'=' * 60}")
        print("TIMING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Mean inference time: {mean_time:.2f} ± {np.std(all_inference_times):.2f} ms")
        print(f"Mean inference rate: {mean_rate:.1f} Hz")

    # Existing summary code...
    all_mse_per_dim = np.array([m["mse_per_dim"] for m in all_metrics])
    mean_mse_per_dim = np.mean(all_mse_per_dim, axis=0)

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Trajectories evaluated: {len(all_metrics)}")
    print(f"Overall MSE: {overall_mse:.6f}")
    print(f"Overall MAE: {overall_mae:.6f}")
    print(f"\nPer-dimension MSE:")
    action_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    for i, (name, mse) in enumerate(zip(action_names, mean_mse_per_dim)):
        print(f"  {name}: {mse:.6f}")

    print(f"\nResults and plots saved to: {output_dir}")


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)
