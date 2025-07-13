import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class SO101Inputs(_transforms.DataTransformFn):
    """Transform inputs from SO101 robot environment to model format."""

    action_dim: int  # This will be 32 from the model config
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Convert tensor to numpy if needed
        front_image = data["observation/images/front"]
        if isinstance(front_image, torch.Tensor):
            front_image = front_image.numpy()

        # Ensure the image is numpy array and handle shape
        front_image = np.asarray(front_image)
        if front_image.ndim == 3 and front_image.shape[-1] != 3:
            if front_image.shape[0] == 3:
                front_image = np.transpose(front_image, (1, 2, 0))

        # Ensure image is uint8 format
        if front_image.dtype != np.uint8:
            if front_image.max() <= 1.0:
                front_image = (front_image * 255).astype(np.uint8)
            else:
                front_image = front_image.astype(np.uint8)

        # Handle state: pad from 6 to 32 dimensions
        so101_state = np.asarray(data["observation/state"])  # Shape: [6]
        state = _transforms.pad_to_dim(so101_state, self.action_dim)  # Pad to [32]

        inputs = {
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": front_image,
                "right_wrist_0_rgb": front_image,
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": True,
            },
            "state": state,  # Now 32-dimensional
            "prompt": data.get("prompt", "Grab the red battery and drop in the box"),
        }

        # Actions are only available during training
        if "actions" in data:
            so101_actions = np.asarray(data["action"])  # Shape: [..., 6]
            # Pad actions from 6 to 32 dimensions
            actions = _transforms.pad_to_dim(so101_actions, self.action_dim)  # Pad to [..., 32]
            inputs["actions"] = actions

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(_transforms.DataTransformFn):
    """Transform model outputs to SO101 robot format."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Extract only the first 6 dimensions (your SO101 robot's DOF)
        full_actions = np.asarray(data["actions"])  # Shape: [..., 32]
        so101_actions = full_actions[..., :6]  # Take only first 6 dimensions

        return {
            "actions": so101_actions,  # Output only 6 DOF actions
        }
