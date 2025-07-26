import dataclasses
import pathlib

from lerobot.datasets import lerobot_dataset
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def debug_inference():
    """Debug what the SO101 model is actually outputting."""

    # Load config and policy
    config_name = "pi0_so101_lora"
    checkpoint_path = "./pi0_lora_so101_checkpoints/so101_experiment/19999"

    train_config = _config.get_config(config_name)
    checkpoint_dir = pathlib.Path(checkpoint_path)

    print("Loading policy...")
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

    print("Policy loaded successfully!")
    print(f"Policy metadata: {policy.metadata}")

    # Load dataset
    dataset = lerobot_dataset.LeRobotDataset("SGPatil/so101_pick_drop")

    # Get first sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")

    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)} = {value}")

    # Create observation with correct key mapping
    # The policy expects the format after repack transforms are applied
    obs_dict = {
        "images": {
            "front": sample["observation.images.front"]  # Note: using the actual key from dataset
        },
        "state": sample["observation.state"],  # Note: using the actual key from dataset
        "prompt": "pick up the object",
    }

    print(f"\nRunning inference...")
    print(f"Input image shape: {obs_dict['images']['front'].shape}")
    print(f"Input state shape: {obs_dict['state'].shape}")
    print(f"Input prompt: {obs_dict['prompt']}")

    result = policy.infer(obs_dict)

    print(f"\nInference result:")
    print(f"Result keys: {result.keys()}")

    for key, value in result.items():
        print(f"  {key}: type={type(value)}")
        if hasattr(value, "shape"):
            print(f"    shape={value.shape}, dtype={value.dtype}")
            print(f"    value={value}")
        else:
            print(f"    value={value}")

    # Try to get actions specifically
    actions = result["actions"]
    print(f"\nActions analysis:")
    print(f"  Type: {type(actions)}")
    if hasattr(actions, "shape"):
        print(f"  Shape: {actions.shape}")
        print(f"  Ndim: {actions.ndim}")
        if actions.ndim > 0:
            print(f"  First few values: {actions.flat[:10]}")

    actions_np = np.asarray(actions)
    print(f"  As numpy - shape: {actions_np.shape}, ndim: {actions_np.ndim}")

    # Compare with ground truth
    gt_action = sample["action"]
    print(f"\nGround truth action:")
    print(f"  Type: {type(gt_action)}")
    print(f"  Shape: {gt_action.shape}")
    print(f"  Values: {gt_action}")


if __name__ == "__main__":
    debug_inference()
