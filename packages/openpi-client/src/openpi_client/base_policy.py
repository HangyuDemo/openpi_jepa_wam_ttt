import abc
from typing import Dict, Optional


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict, *, seed: Optional[int] = None) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
