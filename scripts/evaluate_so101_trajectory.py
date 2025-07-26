import dataclasses
import logging
import pathlib
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

    checkpoint_path: str = "./checkpoints/pi0_so101_lora/my_experiment/19999"
    """Path to the trained model checkpoint."""

    dataset_repo: str = "SGPatil/so101_pick_drop"
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

    action_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

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


def evaluate_single_trajectory(
    policy, dataset, traj_id: int, max_steps: int, action_horizon: int, config: EvalConfig
) -> dict:
    """Evaluate policy on a single trajectory."""

    print(f"Evaluating trajectory {traj_id}...")

    # Get trajectory data
    traj_start_idx = sum(
        dataset.episode_data_index["to"][i] - dataset.episode_data_index["from"][i] for i in range(traj_id)
    )
    traj_length = dataset.episode_data_index["to"][traj_id] - dataset.episode_data_index["from"][traj_id]
    actual_steps = min(max_steps, traj_length)

    predicted_actions = []
    ground_truth_actions = []

    for step in range(actual_steps):
        data_idx = traj_start_idx + step

        try:
            # Get observation data
            sample = dataset[data_idx]

            # Create observation for policy
            obs_dict = {
                "observation/images/front": sample["observation.images.front"],
                "observation/state": sample["observation.state"],
                "prompt": config.dataset_repo.split("/")[-1].replace("_", " "),  # Simple prompt from dataset name
            }

            # Get policy prediction
            result = policy.infer(obs_dict)
            pred_action = result["actions"][0]  # First action from chunk

            # Get ground truth action
            gt_action = sample["action"]

            predicted_actions.append(pred_action)
            ground_truth_actions.append(gt_action)

        except Exception as e:
            print(f"Error at step {step}: {e}")
            break

    if not predicted_actions:
        return {"error": "No valid predictions"}

    # Convert to numpy arrays
    pred_actions = np.array(predicted_actions)
    gt_actions = np.array(ground_truth_actions)

    # Calculate metrics
    metrics = calc_action_mse(pred_actions, gt_actions, action_dim=6)

    # Plot if requested
    if config.plot:
        output_dir = pathlib.Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.save_plots:
            plot_action_trajectory(pred_actions, gt_actions, traj_id, output_dir)
        else:
            plot_action_trajectory(pred_actions, gt_actions, traj_id)

    metrics.update(
        {
            "trajectory_id": traj_id,
            "steps_evaluated": len(predicted_actions),
            "trajectory_length": traj_length,
        }
    )

    return metrics


def main(config: EvalConfig):
    """Main evaluation function."""

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

    num_episodes = len(dataset.episode_data_index["from"])
    actual_trajs = min(config.num_trajectories, num_episodes)

    print(f"Evaluating on {actual_trajs} trajectories...")

    for traj_id in range(actual_trajs):
        metrics = evaluate_single_trajectory(
            policy, dataset, traj_id, config.max_steps_per_traj, config.action_horizon, config
        )

        if "error" not in metrics:
            all_metrics.append(metrics)
            print(
                f"Trajectory {traj_id}: MSE = {metrics['overall_mse']:.4f}, "
                f"MAE = {metrics['mae']:.4f}, Steps = {metrics['steps_evaluated']}"
            )
        else:
            print(f"Trajectory {traj_id}: Failed - {metrics['error']}")

    if not all_metrics:
        print("No successful evaluations!")
        return

    # Aggregate results
    overall_mse = np.mean([m["overall_mse"] for m in all_metrics])
    overall_mae = np.mean([m["mae"] for m in all_metrics])
    overall_max_error = np.mean([m["max_error"] for m in all_metrics])

    # Per-dimension MSE
    all_mse_per_dim = np.array([m["mse_per_dim"] for m in all_metrics])
    mean_mse_per_dim = np.mean(all_mse_per_dim, axis=0)

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Trajectories evaluated: {len(all_metrics)}")
    print(f"Overall MSE: {overall_mse:.6f}")
    print(f"Overall MAE: {overall_mae:.6f}")
    print(f"Overall Max Error: {overall_max_error:.6f}")
    print(f"\nPer-dimension MSE:")
    action_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    for i, (name, mse) in enumerate(zip(action_names, mean_mse_per_dim)):
        print(f"  {name}: {mse:.6f}")

    # Save results
    results_file = output_dir / "evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write("SO101 Policy Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Config: {config.config_name}\n")
        f.write(f"Checkpoint: {config.checkpoint_path}\n")
        f.write(f"Dataset: {config.dataset_repo}\n")
        f.write(f"Trajectories evaluated: {len(all_metrics)}\n\n")
        f.write(f"Overall MSE: {overall_mse:.6f}\n")
        f.write(f"Overall MAE: {overall_mae:.6f}\n")
        f.write(f"Overall Max Error: {overall_max_error:.6f}\n\n")
        f.write("Per-dimension MSE:\n")
        for i, (name, mse) in enumerate(zip(action_names, mean_mse_per_dim)):
            f.write(f"  {name}: {mse:.6f}\n")
        f.write(f"\nDetailed Results:\n")
        for i, metrics in enumerate(all_metrics):
            f.write(
                f"Trajectory {i}: MSE={metrics['overall_mse']:.6f}, "
                f"MAE={metrics['mae']:.6f}, Steps={metrics['steps_evaluated']}\n"
            )

    print(f"\nResults saved to: {results_file}")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)
