import inspect
import logging
import socket
import threading
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from openpi_client import websocket_inference_protocol


_CONNECT_SUPPORTS_PROXY = "proxy" in inspect.signature(websockets.sync.client.connect).parameters


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        connect_timeout: Optional[float] = None,
        retry_interval: Optional[float] = None,
        inference_timeout: Optional[float] = None,
        use_proxy: bool = False,
    ) -> None:
        if connect_timeout is not None and connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive when set")
        if retry_interval is not None and retry_interval < 0:
            raise ValueError("retry_interval must be non-negative when set")
        if inference_timeout is not None and inference_timeout <= 0:
            raise ValueError("inference_timeout must be positive when set")
        if not isinstance(use_proxy, bool):
            raise ValueError("use_proxy must be a boolean")
        if use_proxy and not _CONNECT_SUPPORTS_PROXY:
            raise ValueError("Installed websockets version does not support proxy connections")
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._connect_timeout = connect_timeout
        self._retry_interval = 5.0 if retry_interval is None else retry_interval
        self._inference_timeout = inference_timeout
        self._use_proxy = use_proxy
        self._hard_aborted = False
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = time.monotonic() + self._connect_timeout if self._connect_timeout is not None else None
        while True:
            conn = None
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                connect_kwargs = {
                    "compression": None,
                    "max_size": None,
                    "additional_headers": headers,
                }
                if _CONNECT_SUPPORTS_PROXY and not self._use_proxy:
                    connect_kwargs["proxy"] = None
                if deadline is not None:
                    connect_kwargs["open_timeout"] = self._remaining_timeout(deadline)
                conn = websockets.sync.client.connect(self._uri, **connect_kwargs)
                if deadline is None:
                    metadata_payload = conn.recv()
                else:
                    metadata_payload = conn.recv(timeout=self._remaining_timeout(deadline))
                metadata = msgpack_numpy.unpackb(metadata_payload)
                return conn, metadata
            except ConnectionRefusedError as exc:
                logging.info("Still waiting for server...")
                if deadline is None:
                    time.sleep(self._retry_interval)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._connection_timeout_error() from exc
                time.sleep(min(self._retry_interval, remaining))
            except TimeoutError as exc:
                self._hard_close_connection(conn)
                if deadline is not None:
                    raise self._connection_timeout_error() from exc
                raise
            except Exception:
                self._close_connection(conn)
                raise

    def _remaining_timeout(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._connection_timeout_error()
        return remaining

    def _connection_timeout_error(self) -> TimeoutError:
        return TimeoutError(f"Timed out waiting for policy server at {self._uri} after {self._connect_timeout} seconds")

    @staticmethod
    def _close_connection(conn: Optional[websockets.sync.client.ClientConnection]) -> None:
        if conn is not None:
            conn.close()

    @staticmethod
    def _hard_close_connection(conn: Optional[websockets.sync.client.ClientConnection]) -> None:
        if conn is None:
            return
        close_socket = getattr(conn, "close_socket", None)
        if callable(close_socket):
            try:
                close_socket()
                return
            except Exception:
                logging.exception("Failed to close websocket transport directly")
        raw_socket = getattr(conn, "socket", None)
        if raw_socket is None:
            return
        try:
            raw_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            raw_socket.close()
        except OSError:
            pass

    def _abort_connection(self) -> None:
        self._hard_aborted = True
        self._hard_close_connection(self._ws)

    def _inference_timeout_error(self) -> TimeoutError:
        return TimeoutError(
            f"Timed out sending policy request or waiting for response from {self._uri} "
            f"after {self._inference_timeout} seconds"
        )

    def _remaining_inference_timeout(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._inference_timeout_error()
        return remaining

    @override
    def infer(self, obs: Dict, *, seed: Optional[int] = None) -> Dict:  # noqa: UP006
        if seed is not None and not websocket_inference_protocol.supports_seeded_inference(self._server_metadata):
            raise RuntimeError("Policy server does not advertise seeded inference support")
        request = websocket_inference_protocol.make_inference_request(obs, seed)
        data = self._packer.pack(request)
        deadline = time.monotonic() + self._inference_timeout if self._inference_timeout is not None else None
        completed = threading.Event()
        timed_out = threading.Event()

        def abort_on_timeout() -> None:
            if completed.is_set():
                return
            timed_out.set()
            self._abort_connection()

        timer = None
        if self._inference_timeout is not None:
            timer = threading.Timer(self._inference_timeout, abort_on_timeout)
            timer.daemon = True
            timer.start()
        try:
            self._ws.send(data)
            if deadline is None:
                response = self._ws.recv()
            else:
                response = self._ws.recv(timeout=self._remaining_inference_timeout(deadline))
        except Exception as exc:
            if deadline is not None and (timed_out.is_set() or isinstance(exc, TimeoutError)):
                if not timed_out.is_set():
                    timed_out.set()
                    self._abort_connection()
                raise self._inference_timeout_error() from exc
            raise
        finally:
            completed.set()
            if timer is not None:
                timer.cancel()
        if timed_out.is_set():
            raise self._inference_timeout_error()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass

    def close(self) -> None:
        if not self._hard_aborted:
            self._close_connection(self._ws)
