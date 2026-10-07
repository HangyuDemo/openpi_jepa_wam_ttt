import contextlib
import threading
import time
from unittest import mock

import pytest

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
from openpi_client import websocket_inference_protocol


def test_default_connection_behavior_is_backward_compatible():
    connection = mock.Mock()
    connection.recv.return_value = msgpack_numpy.packb({"ready": True})

    with mock.patch.object(
        websocket_client_policy.websockets.sync.client,
        "connect",
        return_value=connection,
    ) as connect:
        policy = websocket_client_policy.WebsocketClientPolicy("localhost", 8000)

    expected_kwargs = {"compression": None, "max_size": None, "additional_headers": None}
    if websocket_client_policy._CONNECT_SUPPORTS_PROXY:
        expected_kwargs["proxy"] = None
    connect.assert_called_once_with("ws://localhost:8000", **expected_kwargs)
    connection.recv.assert_called_once_with()
    assert policy.get_server_metadata() == {"ready": True}


def test_connection_deadline_bounds_open_and_metadata_receive():
    connection = mock.Mock()
    connection.recv.return_value = msgpack_numpy.packb({})

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(websocket_client_policy.time, "monotonic", side_effect=[100.0, 100.25, 100.5])
        )
        connect = stack.enter_context(
            mock.patch.object(
                websocket_client_policy.websockets.sync.client,
                "connect",
                return_value=connection,
            )
        )
        websocket_client_policy.WebsocketClientPolicy(
            "localhost",
            8000,
            connect_timeout=2.0,
            retry_interval=0.1,
        )

    assert connect.call_args.kwargs["open_timeout"] == pytest.approx(1.75)
    connection.recv.assert_called_once_with(timeout=pytest.approx(1.5))


def test_connection_refusal_stops_at_deadline():
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                websocket_client_policy.time,
                "monotonic",
                side_effect=[10.0, 10.0, 10.75, 11.1],
            )
        )
        sleep = stack.enter_context(mock.patch.object(websocket_client_policy.time, "sleep"))
        connect = stack.enter_context(
            mock.patch.object(
                websocket_client_policy.websockets.sync.client,
                "connect",
                side_effect=ConnectionRefusedError("refused"),
            )
        )
        with pytest.raises(TimeoutError, match="Timed out waiting for policy server"):
            websocket_client_policy.WebsocketClientPolicy(
                "localhost",
                8000,
                connect_timeout=1.0,
                retry_interval=0.25,
            )

    assert connect.call_count == 1
    sleep.assert_called_once_with(0.25)


def test_metadata_timeout_closes_connection():
    connection = mock.Mock()
    connection.recv.side_effect = TimeoutError("metadata stalled")

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(websocket_client_policy.time, "monotonic", side_effect=[1.0, 1.1, 1.2]))
        stack.enter_context(
            mock.patch.object(
                websocket_client_policy.websockets.sync.client,
                "connect",
                return_value=connection,
            )
        )
        with pytest.raises(TimeoutError, match="Timed out waiting for policy server"):
            websocket_client_policy.WebsocketClientPolicy("localhost", 8000, connect_timeout=2.0)

    connection.close_socket.assert_called_once_with()
    connection.close.assert_not_called()


def test_inference_timeout_hard_closes_stalled_receive():
    connection = mock.Mock()
    connection.recv.side_effect = [msgpack_numpy.packb({"ready": True}), TimeoutError("inference stalled")]

    with mock.patch.object(
        websocket_client_policy.websockets.sync.client,
        "connect",
        return_value=connection,
    ):
        policy = websocket_client_policy.WebsocketClientPolicy(
            "localhost",
            8000,
            inference_timeout=3.0,
        )
        with pytest.raises(TimeoutError, match="Timed out sending policy request or waiting for response"):
            policy.infer({"state": [0.0]})

    assert connection.recv.call_args_list[0] == mock.call()
    assert 0 < connection.recv.call_args_list[1].kwargs["timeout"] <= 3.0
    connection.close_socket.assert_called_once_with()
    connection.close.assert_not_called()
    policy.close()
    connection.close.assert_not_called()


def test_inference_timeout_hard_closes_stalled_send_without_graceful_close():
    class StalledConnection:
        def __init__(self):
            self._metadata_sent = False
            self._released = threading.Event()
            self.hard_close_calls = 0
            self.graceful_close_calls = 0

        def recv(self, timeout=None):
            del timeout
            if self._metadata_sent:
                raise AssertionError("recv must not be reached while send is stalled")
            self._metadata_sent = True
            return msgpack_numpy.packb({"ready": True})

        def send(self, _data):
            self._released.wait(timeout=1.0)
            raise OSError("socket was aborted")

        def close_socket(self):
            self.hard_close_calls += 1
            self._released.set()

        def close(self):
            self.graceful_close_calls += 1

    connection = StalledConnection()
    with mock.patch.object(
        websocket_client_policy.websockets.sync.client,
        "connect",
        return_value=connection,
    ):
        policy = websocket_client_policy.WebsocketClientPolicy(
            "localhost",
            8000,
            inference_timeout=0.01,
        )
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="Timed out sending policy request or waiting for response"):
            policy.infer({"state": [0.0]})
        elapsed = time.monotonic() - started

    policy.close()
    assert elapsed < 0.5
    assert connection.hard_close_calls == 1
    assert connection.graceful_close_calls == 0


def test_send_time_is_deducted_from_receive_deadline():
    connection = mock.Mock()
    connection.recv.side_effect = [msgpack_numpy.packb({"ready": True}), msgpack_numpy.packb({"actions": []})]

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(websocket_client_policy.time, "monotonic", side_effect=[10.0, 10.75]))
        stack.enter_context(
            mock.patch.object(
                websocket_client_policy.websockets.sync.client,
                "connect",
                return_value=connection,
            )
        )
        policy = websocket_client_policy.WebsocketClientPolicy("localhost", 8000, inference_timeout=1.0)
        policy.infer({"state": [0.0]})

    connection.recv.assert_has_calls([mock.call(), mock.call(timeout=pytest.approx(0.25))])


def test_seeded_inference_uses_versioned_request_envelope():
    metadata = websocket_inference_protocol.with_server_capabilities({"ready": True})
    connection = mock.Mock()
    connection.recv.side_effect = [msgpack_numpy.packb(metadata), msgpack_numpy.packb({"actions": [1.0]})]

    with mock.patch.object(
        websocket_client_policy.websockets.sync.client,
        "connect",
        return_value=connection,
    ):
        policy = websocket_client_policy.WebsocketClientPolicy("localhost", 8000)
        response = policy.infer({"state": [0.0]}, seed=123)

    packed_request = connection.send.call_args.args[0]
    observation, seed = websocket_inference_protocol.parse_inference_request(msgpack_numpy.unpackb(packed_request))
    assert observation == {"state": [0.0]}
    assert seed == 123
    assert response == {"actions": [1.0]}


def test_seeded_inference_rejects_legacy_server_before_send():
    connection = mock.Mock()
    connection.recv.return_value = msgpack_numpy.packb({"ready": True})

    with mock.patch.object(
        websocket_client_policy.websockets.sync.client,
        "connect",
        return_value=connection,
    ):
        policy = websocket_client_policy.WebsocketClientPolicy("localhost", 8000)
        with pytest.raises(RuntimeError, match="does not advertise"):
            policy.infer({"state": [0.0]}, seed=123)

    connection.send.assert_not_called()


def test_proxy_is_disabled_when_supported_by_websockets():
    if not websocket_client_policy._CONNECT_SUPPORTS_PROXY:
        pytest.skip("Installed websockets has no proxy argument")
    connection = mock.Mock()
    connection.recv.return_value = msgpack_numpy.packb({})

    with mock.patch.object(
        websocket_client_policy.websockets.sync.client,
        "connect",
        return_value=connection,
    ) as connect:
        websocket_client_policy.WebsocketClientPolicy("localhost", 8000)

    assert connect.call_args.kwargs["proxy"] is None
