"""Bridge RoboTwin 2.0 XPolicyLab observations to an OpenPI policy server."""

from collections.abc import Mapping

import numpy as np
from openpi_client import websocket_client_policy
from openpi_client import websocket_inference_protocol


def _rgb(vision: Mapping, name: str) -> np.ndarray:
    camera = vision[name]
    image = np.asarray(camera["color"] if isinstance(camera, Mapping) else camera)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be an HWC RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"{name} must have uint8 or floating RGB pixels, got {image.dtype}")
    return image


def encode_observation(obs: Mapping) -> dict:
    """XPolicyLab's joint state order matches the local 14D LeRobot conversion."""
    state = obs["state"]
    parts = [
        np.asarray(state["left_arm_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(state["left_ee_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(state["right_arm_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(state["right_ee_joint_state"], dtype=np.float32).reshape(-1),
    ]
    if [part.size for part in parts] != [6, 1, 6, 1]:
        raise ValueError(f"Expected Aloha-Agilex joint dimensions [6, 1, 6, 1], got {[p.size for p in parts]}")
    vector = np.concatenate(parts)
    if not np.isfinite(vector).all():
        raise ValueError("RoboTwin state contains non-finite values")
    vision = obs["vision"]
    return {
        "state": vector,
        "images": {
            "cam_high": _rgb(vision, "cam_head"),
            "cam_left_wrist": _rgb(vision, "cam_left_wrist"),
            "cam_right_wrist": _rgb(vision, "cam_right_wrist"),
        },
        "prompt": str(obs["instruction"]),
    }


class OpenPIModelClient:
    """Implements the subset of XPolicyLab's ModelClient used by RoboTwin."""

    _robotwin_protocol = "ws"

    def __init__(self, host: str, port: int, replan_steps: int, *, timeout: float = 300):
        if replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port, connect_timeout=30, inference_timeout=timeout, use_proxy=False
        )
        metadata = self.policy.get_server_metadata()
        if not websocket_inference_protocol.supports_reset(metadata):
            raise ValueError("Policy server lacks episode reset support; restart the server with this checkout")
        horizon = metadata.get("action_horizon")
        if horizon is not None and replan_steps > int(horizon):
            raise ValueError(f"replan_steps={replan_steps} exceeds model action_horizon={horizon}")
        stride = metadata.get("ttt_observation_stride")
        if stride is not None and replan_steps != int(stride):
            raise ValueError(f"TTT checkpoint expects replan_steps={stride}, got {replan_steps}")
        if metadata.get("ttt_memory_source") == "explicit_jepa":
            raise ValueError("Explicit-JEPA TTT requires an online V-JEPA encoder; this adapter does not supply it")
        self.replan_steps = replan_steps
        self.observation = None
        self.seed = None
        self.inference_index = 0

    def call(self, func_name: str, obs=None):
        if func_name == "prepare_case":
            self.seed = int(obs["seed"]) % (2**32)
            return None
        if func_name == "reset":
            self.policy.reset()
            self.observation = None
            self.inference_index = 0
            return None
        if func_name == "update_obs":
            self.observation = encode_observation(obs)
            return None
        if func_name == "get_action":
            if self.observation is None:
                raise RuntimeError("update_obs must be called before get_action")
            seed = None if self.seed is None else (self.seed + self.inference_index) % (2**32)
            result = self.policy.infer(self.observation, seed=seed)
            self.inference_index += 1
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < self.replan_steps:
                raise ValueError(f"Expected at least {self.replan_steps} x 14 actions, got {actions.shape}")
            actions = actions[: self.replan_steps]
            if not np.isfinite(actions).all():
                raise ValueError("Policy returned non-finite actions")
            return actions
        if func_name == "trial_end":
            return None
        raise ValueError(f"Unsupported RoboTwin policy call: {func_name}")

    def close(self):
        self.policy.close()
