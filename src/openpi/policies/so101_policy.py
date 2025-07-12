import dataclasses
from typing import Any

import jax.numpy as jnp
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.transforms as _transforms

@dataclasses.dataclass(frozen=True)
class SO101Inputs(_transforms.Transform):
    """Transform inputs from SO101 robot environment to model format."""
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Map your SO101 dataset keys to model expected keys
        return {
            "images": {
                # Adjust these camera names to match your dataset
                "base_0_rgb": data["observation/image"],  # or whatever your main camera is called
                "left_wrist_0_rgb": data.get("observation/wrist_image", data["observation/image"]),  # adjust as needed
                "right_wrist_0_rgb": data.get("observation/wrist_image", data["observation/image"]),  # adjust as needed
            },
            "image_masks": {
                "base_0_rgb": jnp.ones(data["observation/image"].shape[0], dtype=bool),
                "left_wrist_0_rgb": jnp.ones(data["observation/image"].shape[0], dtype=bool),
                "right_wrist_0_rgb": jnp.ones(data["observation/image"].shape[0], dtype=bool),
            },
            "state": data["observation/state"],
            "actions": data["actions"],
            "prompt": data.get("prompt", "Control the SO101 robot arm"),
        }

@dataclasses.dataclass(frozen=True)
class SO101Outputs(_transforms.Transform):
    """Transform model outputs to SO101 robot format."""
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "actions": data["actions"],
        }