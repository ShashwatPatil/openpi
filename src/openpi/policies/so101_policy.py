import dataclasses
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class SO101Inputs(_transforms.DataTransformFn):
    """Transform inputs from SO101 robot environment to model format."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Based on your dataset structure, you have:
        # - observation.images.front (your camera)
        # - observation.state (robot joint states)
        # - action (robot actions)

        # Map to the expected format for π₀ model
        return {
            "images": {
                # Use your front camera for all camera views (π₀ expects 3 cameras)
                "base_0_rgb": data["observation/images/front"],
                "left_wrist_0_rgb": data["observation/images/front"],  # Duplicate front camera
                "right_wrist_0_rgb": data["observation/images/front"],  # Duplicate front camera
            },
            "image_masks": {
                "base_0_rgb": jnp.ones(data["observation/images/front"].shape[0], dtype=bool),
                "left_wrist_0_rgb": jnp.ones(data["observation/images/front"].shape[0], dtype=bool),
                "right_wrist_0_rgb": jnp.ones(data["observation/images/front"].shape[0], dtype=bool),
            },
            "state": data["observation/state"],
            "actions": data["action"],
            "prompt": data.get("prompt", "Grab the red battery and drop in the box"),
        }


@dataclasses.dataclass(frozen=True)
class SO101Outputs(_transforms.DataTransformFn):
    """Transform model outputs to SO101 robot format."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "actions": data["actions"],
        }
