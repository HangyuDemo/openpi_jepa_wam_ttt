import pytest

from openpi_client import websocket_inference_protocol


def test_seeded_request_round_trip_and_capability():
    metadata = websocket_inference_protocol.with_server_capabilities({"model": "pi05"})
    observation = {"state": [1.0, 2.0]}
    request = websocket_inference_protocol.make_inference_request(observation, 123)

    assert websocket_inference_protocol.supports_seeded_inference(metadata)
    assert websocket_inference_protocol.parse_inference_request(request) == (observation, 123)
    assert websocket_inference_protocol.parse_inference_request(observation) == (observation, None)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "1"])
def test_seeded_request_rejects_non_uint32_seed(seed):
    with pytest.raises(ValueError, match="uint32"):
        websocket_inference_protocol.make_inference_request({}, seed)


def test_seeded_request_rejects_ambiguous_or_malformed_envelope():
    key = websocket_inference_protocol.SEEDED_INFERENCE_REQUEST_KEY
    valid = websocket_inference_protocol.make_inference_request({}, 1)

    with pytest.raises(ValueError, match="only the protocol envelope"):
        websocket_inference_protocol.parse_inference_request({**valid, "extra": True})
    with pytest.raises(ValueError, match="Malformed"):
        websocket_inference_protocol.parse_inference_request({key: {"seed": 1}})
