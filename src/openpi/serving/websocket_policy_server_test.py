# ruff: noqa: SLF001

import multiprocessing
import socket
import time

from openpi_client import websocket_client_policy
from openpi_client import websocket_inference_protocol
import pytest

from openpi.serving import websocket_policy_server


class _FakePolicy:
    def __init__(self):
        self.calls = []

    def infer(self, observation, *, seed=None):
        self.calls.append((observation, seed))
        return {"actions": [seed]}


def _serve_fake_policy(port):
    websocket_policy_server.WebsocketPolicyServer(_FakePolicy(), host="127.0.0.1", port=port).serve_forever()


class _StalledPolicy:
    def infer(self, observation, *, seed=None):
        del observation, seed
        time.sleep(30)
        return {"actions": []}


def _serve_stalled_policy(port):
    websocket_policy_server.WebsocketPolicyServer(_StalledPolicy(), host="127.0.0.1", port=port).serve_forever()


def _unused_local_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_infer_request_preserves_legacy_and_seeded_behavior():
    policy = _FakePolicy()
    observation = {"state": [0.0]}

    assert websocket_policy_server._infer_request(policy, observation) == {"actions": [None]}
    request = websocket_inference_protocol.make_inference_request(observation, 123)
    assert websocket_policy_server._infer_request(policy, request) == {"actions": [123]}
    assert policy.calls == [(observation, None), (observation, 123)]


def test_server_metadata_advertises_seeded_inference():
    server = websocket_policy_server.WebsocketPolicyServer(_FakePolicy(), metadata={"model": "pi05"})

    assert websocket_inference_protocol.supports_seeded_inference(server._metadata)
    assert server._metadata["model"] == "pi05"


def test_live_websocket_seeded_inference_round_trip():
    port = _unused_local_port()
    process = multiprocessing.get_context("fork").Process(target=_serve_fake_policy, args=(port,))
    process.start()
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1",
            port,
            connect_timeout=5.0,
            retry_interval=0.05,
            inference_timeout=2.0,
        )
        response = client.infer({"state": [0.0]}, seed=123)
        assert response["actions"] == [123]
        assert "server_timing" in response
        client._ws.close()
    finally:
        process.terminate()
        process.join(timeout=5.0)
    assert process.exitcode is not None


def test_live_websocket_response_stall_honors_hard_deadline():
    port = _unused_local_port()
    process = multiprocessing.get_context("fork").Process(target=_serve_stalled_policy, args=(port,))
    process.start()
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1",
            port,
            connect_timeout=5.0,
            retry_interval=0.05,
            inference_timeout=0.2,
        )
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="Timed out sending policy request or waiting for response"):
            client.infer({"state": [0.0]}, seed=123)
        assert time.monotonic() - started < 1.0

        close_started = time.monotonic()
        client.close()
        assert time.monotonic() - close_started < 0.1
    finally:
        process.terminate()
        process.join(timeout=5.0)
    assert process.exitcode is not None
