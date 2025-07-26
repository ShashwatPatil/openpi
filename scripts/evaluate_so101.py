import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def evaluate_model(checkpoint_path: str, config_name: str = "pi0_so101_lora", num_eval_batches: int = 100):
    """Evaluate the trained model on validation data."""

    # Load config and model
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, batch_size=1)  # Use small batch size

    params = _model.restore_params(checkpoint_path)

    model = config.model.load(params)

    # Create data loader (using same dataset for now, but you could split train/val)
    data_loader = _data_loader.create_data_loader(
        config,
        skip_norm_stats=False,  # Use norm stats if available
        num_batches=num_eval_batches,
        shuffle=False,  # Don't shuffle for evaluation
    )

    # Evaluate
    losses = []
    rng = jax.random.key(42)

    print(f"Evaluating model on {num_eval_batches} batches...")

    for i, (obs, actions) in enumerate(data_loader):
        rng, eval_rng = jax.random.split(rng)
        loss = model.compute_loss(eval_rng, obs, actions, train=False)
        losses.append(np.asarray(loss))

        if i % 10 == 0:
            print(f"Batch {i}/{num_eval_batches}, Loss: {np.mean(loss):.4f}")

    # Compute statistics
    all_losses = np.concatenate(losses)
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)

    print(f"\nEvaluation Results:")
    print(f"Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Min Loss: {np.min(all_losses):.4f}")
    print(f"Max Loss: {np.max(all_losses):.4f}")

    return {
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "min_loss": np.min(all_losses),
        "max_loss": np.max(all_losses),
        "all_losses": all_losses,
    }


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "../pi0_lora_so101_checkpoints/so101_experiment/19999/params"  # Adjust path
    results = evaluate_model(checkpoint_path)
