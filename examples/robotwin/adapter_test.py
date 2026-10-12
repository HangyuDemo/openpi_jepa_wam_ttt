from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import adapter


def _obs():
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    return {
        "instruction": "beat the block with the hammer",
        "vision": {name: {"color": image} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
        "state": {
            "left_arm_joint_state": np.arange(6),
            "left_ee_joint_state": np.array([0.2]),
            "right_arm_joint_state": np.arange(6) + 10,
            "right_ee_joint_state": np.array([0.8]),
        },
    }


def test_observation_mapping_uses_training_camera_and_joint_order():
    encoded = adapter.encode_observation(_obs())
    np.testing.assert_allclose(encoded["state"], [0, 1, 2, 3, 4, 5, 0.2, 10, 11, 12, 13, 14, 15, 0.8])
    assert set(encoded["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert encoded["prompt"] == "beat the block with the hammer"


def test_action_chunk_replans_and_resets_between_episodes():
    policy = mock.Mock()
    policy.get_server_metadata.return_value = {
        "action_horizon": 50,
        "ttt_observation_stride": 8,
        "__openpi_server_capabilities__": {"reset": 1},
    }
    policy.infer.return_value = {"actions": np.ones((50, 14), dtype=np.float32)}
    with mock.patch.object(adapter.websocket_client_policy, "WebsocketClientPolicy", return_value=policy):
        client = adapter.OpenPIModelClient("localhost", 8000, 8)
    client.call("prepare_case", {"seed": 123})
    client.call("reset")
    client.call("update_obs", _obs())
    assert client.call("get_action").shape == (8, 14)
    assert policy.infer.call_args.kwargs["seed"] == 123
    client.call("get_action")
    assert policy.infer.call_args.kwargs["seed"] == 124
    client.call("reset")
    client.call("update_obs", _obs())
    client.call("get_action")
    assert policy.infer.call_args.kwargs["seed"] == 123
    assert policy.reset.call_count == 2


def test_ttt_stride_mismatch_rejected_before_rollout():
    policy = mock.Mock()
    policy.get_server_metadata.return_value = {
        "action_horizon": 50,
        "ttt_observation_stride": 8,
        "__openpi_server_capabilities__": {"reset": 1},
    }
    with (
        mock.patch.object(adapter.websocket_client_policy, "WebsocketClientPolicy", return_value=policy),
        pytest.raises(ValueError, match="expects replan_steps=8"),
    ):
        adapter.OpenPIModelClient("localhost", 8000, 16)
