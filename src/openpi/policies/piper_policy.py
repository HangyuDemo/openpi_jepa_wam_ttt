"""Piper single-arm joint-position inputs for Pi0.5 training and inference."""

import dataclasses

import numpy as np

from openpi import transforms


def _image_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D RGB image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map 14D state, 7D joint commands, and front/wrist cameras to Pi0.5."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != 14:
            raise ValueError(f"Piper state must have 14 dimensions, got {state.shape}")
        front = _image_to_hwc_uint8(data["images"]["front"])
        wrist = _image_to_hwc_uint8(data["images"]["wrist"])
        result = {
            "state": state,
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 7:
                raise ValueError(f"Piper action must have 7 dimensions, got {actions.shape}")
            result["actions"] = actions.copy()
        for key in ("prompt", "vjepa_target", "vjepa_current"):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Return the robot's seven joint/gripper commands from padded actions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}
