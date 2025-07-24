import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        base_image = np.asarray(data["images"]["front"])

        images = { "base": base_image }
        image_masks = { "base": np.True_ }
        
        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "task" in data:
            inputs["prompt"] = data["task"]
        
        return inputs
    
@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:,:6])
        return {"actions": actions}