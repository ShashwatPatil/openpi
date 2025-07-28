import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Parse and validate image format."""
    image = np.asarray(image)

    # Handle video data (if it's encoded)
    if image.ndim == 1:
        # This might be encoded video data - you'll need to decode it
        raise ValueError(
            f"Got 1D image data with shape {image.shape}. This might be encoded video data that needs decoding."
        )

    # Convert to uint8 if it's float
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = (255 * image).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Ensure we have a proper image shape (H, W, C)
    if image.ndim == 3 and image.shape[0] == 3:
        # Convert from CHW to HWC
        image = np.transpose(image, (1, 2, 0))
    elif image.ndim != 3 or image.shape[2] not in [1, 3, 4]:
        raise ValueError(f"Invalid image shape: {image.shape}. Expected (H, W, C) where C is 1, 3, or 4.")

    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Parse and validate the image
        front_image = _parse_image(data["images"]["front"])
        laptop_image = _parse_image(data["images"]["laptop"])

        # Provide all expected camera views (duplicate the single camera for missing views)
        images = {
            "front": front_image,  # Main camera
            "laptop": laptop_image,  # Duplicate for left wrist camera
        }
        image_masks = {
            "front": np.True_,
            "laptop": np.True_,
        }

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :6])
        return {"actions": actions}
