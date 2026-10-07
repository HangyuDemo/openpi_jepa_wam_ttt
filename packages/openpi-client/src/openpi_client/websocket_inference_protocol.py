from typing import Any, Dict, Optional, Tuple


SERVER_CAPABILITIES_KEY = "__openpi_server_capabilities__"
SEEDED_INFERENCE_CAPABILITY = "seeded_inference"
SEEDED_INFERENCE_VERSION = 1
SEEDED_INFERENCE_REQUEST_KEY = "__openpi_seeded_inference_request__"
_UINT32_MAX = 2**32 - 1


def with_server_capabilities(metadata: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(metadata)
    raw_capabilities = result.get(SERVER_CAPABILITIES_KEY)
    capabilities = dict(raw_capabilities) if isinstance(raw_capabilities, dict) else {}
    capabilities[SEEDED_INFERENCE_CAPABILITY] = SEEDED_INFERENCE_VERSION
    result[SERVER_CAPABILITIES_KEY] = capabilities
    return result


def supports_seeded_inference(metadata: Dict[str, Any]) -> bool:
    capabilities = metadata.get(SERVER_CAPABILITIES_KEY)
    return isinstance(capabilities, dict) and capabilities.get(SEEDED_INFERENCE_CAPABILITY) == SEEDED_INFERENCE_VERSION


def make_inference_request(observation: Dict[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    if seed is None:
        return observation
    _validate_seed(seed)
    return {
        SEEDED_INFERENCE_REQUEST_KEY: {
            "version": SEEDED_INFERENCE_VERSION,
            "seed": seed,
            "observation": observation,
        }
    }


def parse_inference_request(request: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[int]]:
    if SEEDED_INFERENCE_REQUEST_KEY not in request:
        return request, None
    if set(request) != {SEEDED_INFERENCE_REQUEST_KEY}:
        raise ValueError("Seeded inference request must contain only the protocol envelope")
    envelope = request[SEEDED_INFERENCE_REQUEST_KEY]
    if not isinstance(envelope, dict) or set(envelope) != {"version", "seed", "observation"}:
        raise ValueError("Malformed seeded inference request envelope")
    if envelope["version"] != SEEDED_INFERENCE_VERSION:
        raise ValueError(f"Unsupported seeded inference protocol version: {envelope['version']!r}")
    seed = envelope["seed"]
    _validate_seed(seed)
    observation = envelope["observation"]
    if not isinstance(observation, dict):
        raise ValueError("Seeded inference observation must be a mapping")
    return observation, seed


def _validate_seed(seed: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _UINT32_MAX:
        raise ValueError(f"Inference seed must be a uint32, got {seed!r}")
