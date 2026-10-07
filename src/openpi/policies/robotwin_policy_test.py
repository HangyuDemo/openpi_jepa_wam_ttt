import numpy as np

from openpi import transforms
from openpi.policies import robotwin_policy


def _sample(state, actions):
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    return {
        "state": state,
        "actions": actions,
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "prompt": "move the block",
    }


def test_checkpoint_gripper_values_and_joint_signs():
    state = np.zeros(14, dtype=np.float32)
    state[[1, 2, 8, 9]] = [0.3, -0.4, 0.5, -0.6]
    actions = np.stack([state.copy(), state.copy()])
    state[[6, 13]] = [0, 1]
    actions[:, [6, 13]] = [[0, 1], [1, 0]]

    result = robotwin_policy.RoboTwinInputs()(_sample(state, actions))

    np.testing.assert_allclose(result["state"][[1, 2, 8, 9]], [-0.3, 0.4, -0.5, 0.6])
    np.testing.assert_allclose(result["state"][[6, 13]], [-1.07986555, 0.9456528], atol=1e-5)
    np.testing.assert_allclose(
        result["actions"][:, [6, 13]], [[-1.1689, 0.9434], [0.9434, -1.1689]], atol=1e-5
    )
    np.testing.assert_array_equal(result["image"]["base_0_rgb"].shape, (8, 8, 3))


def test_delta_and_pi_transforms_round_trip_to_robotwin_actions():
    state = np.array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.0] * 2, dtype=np.float32)
    actions = np.stack([state.copy(), state.copy()])
    actions[0, [0, 1, 7, 8, 6, 13]] = [0.2, 0.3, 0.5, -0.1, 1, 1]
    actions[1, [0, 1, 7, 8, 6, 13]] = [-0.2, -0.3, 0.6, 0.1, 0, 1]
    sample = _sample(state, actions)
    mask = transforms.make_bool_mask(6, -1, 6, -1)

    encoded = transforms.DeltaActions(mask)(robotwin_policy.RoboTwinInputs()(sample))
    decoded = transforms.AbsoluteActions(mask)({"state": encoded["state"], "actions": encoded["actions"].copy()})
    result = robotwin_policy.RoboTwinOutputs()(decoded)

    np.testing.assert_allclose(result["actions"], actions, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sample["state"], state)
    np.testing.assert_array_equal(sample["actions"], actions)


def test_raw_mode_preserves_values():
    state = np.arange(14, dtype=np.float32)
    actions = np.stack([state, state + 1])
    encoded = robotwin_policy.RoboTwinInputs(adapt_to_pi=False)(_sample(state, actions))
    result = robotwin_policy.RoboTwinOutputs(adapt_to_pi=False)({"actions": encoded["actions"]})

    np.testing.assert_array_equal(encoded["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)
