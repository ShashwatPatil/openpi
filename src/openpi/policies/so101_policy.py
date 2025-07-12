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

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Based on your dataset structure from info.json:
        # - observation.images.front (your camera)
        # - observation.state (robot joint states)
        # - action (robot actions)

        # Get the front image and ensure proper format
        front_image = data["observation/images/front"]

        # Convert tensor to numpy if needed and ensure proper shape
        if isinstance(front_image, torch.Tensor):
            front_image = front_image.numpy()

        # Ensure the image is numpy array
        front_image = np.asarray(front_image)

        # Debug: print image info
        print(f"DEBUG: front_image shape: {front_image.shape}, dtype: {front_image.dtype}")

        # Expected shape from info.json: [480, 640, 3]
        # If we have a different shape, we need to handle it
        if front_image.ndim == 4 and front_image.shape[0] == 1:
            # Remove batch dimension if present: (1, H, W, C) -> (H, W, C)
            front_image = front_image[0]
        elif front_image.ndim == 3 and front_image.shape[-1] != 3:
            # If shape is (C, H, W), transpose to (H, W, C)
            if front_image.shape[0] == 3:
                front_image = np.transpose(front_image, (1, 2, 0))

        # Ensure image is uint8 format (PIL expects this)
        if front_image.dtype != np.uint8:
            # If float image, convert to uint8 (assuming values are in [0, 1] or [0, 255])
            if front_image.max() <= 1.0:
                front_image = (front_image * 255).astype(np.uint8)
            else:
                front_image = front_image.astype(np.uint8)

        # DON'T add batch dimension here - the dataloader will handle batching
        # The image should be (H, W, C) not (1, H, W, C)

        print(f"DEBUG: final front_image shape: {front_image.shape}, dtype: {front_image.dtype}")

        # Map to the expected format for π₀ model
        return {
            "image": {
                # Use your front camera for all camera views (π₀ expects 3 cameras)
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": front_image,  # Duplicate front camera
                "right_wrist_0_rgb": front_image,  # Duplicate front camera
            },
            "image_mask": {
                # These should be scalar booleans, not arrays with batch dimension
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": True,
            },
            "state": data["observation/state"],
            "actions": data["action"],  # Map dataset's "action" to expected "actions"
            "prompt": data.get("prompt", "Grab the red battery and drop in the box"),
        }


@dataclasses.dataclass(frozen=True)
class SO101Outputs(_transforms.DataTransformFn):
    """Transform model outputs to SO101 robot format."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "actions": data["actions"],  # Output "actions" (plural)
        }
