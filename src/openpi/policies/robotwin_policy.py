"""RoboTwin 2.0 joint-space inputs for Pi0.5 training and inference."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import aloha_policy


def _image_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D RGB image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    """Map three RoboTwin cameras and raw 14D joint state/actions to Pi0.5 keys."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] != 14:
            raise ValueError(f"RoboTwin state must have 14 dimensions, got {state.shape}")
        state = aloha_policy._decode_state(state, adapt_to_pi=self.adapt_to_pi)

        cameras = data["images"]
        images = {
            "base_0_rgb": _image_to_hwc_uint8(cameras["cam_high"]),
            "left_wrist_0_rgb": _image_to_hwc_uint8(cameras["cam_left_wrist"]),
            "right_wrist_0_rgb": _image_to_hwc_uint8(cameras["cam_right_wrist"]),
        }
        result = {
            "image": images,
            "image_mask": {name: np.True_ for name in images},
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 14:
                raise ValueError(f"RoboTwin actions must have 14 dimensions, got {actions.shape}")
            result["actions"] = aloha_policy._encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        if "vjepa_target" in data:
            result["vjepa_target"] = data["vjepa_target"]
        if "vjepa_current" in data:
            result["vjepa_current"] = data["vjepa_current"]
        return result


@dataclasses.dataclass(frozen=True)
class RoboTwinOutputs(transforms.DataTransformFn):
    """Convert Pi0.5 actions back to RoboTwin's joint and gripper units."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])[..., :14]
        return {"actions": aloha_policy._encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
