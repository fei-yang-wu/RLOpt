"""Expert replay buffer interface for imitation learning algorithms.

This module provides a unified interface for expert demonstration replay buffers
that can wrap different backends, with first-class support for ImitationLearningTools'
ExpertReplayManager.
"""

from __future__ import annotations

from collections.abc import Sized
from typing import Any, Protocol, runtime_checkable

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer


@runtime_checkable
class ExpertBufferProtocol(Protocol):
    """Protocol for expert replay buffer interface.

    Any object implementing this protocol can be used as an expert data source
    for imitation learning algorithms.
    """

    def sample(self) -> TensorDict:
        """Sample a batch of expert demonstrations.

        Returns:
            TensorDict with keys:
                - 'observation': state at time t
                - 'action': action at time t
                - ('next', 'observation'): state at time t+1
                - Additional keys may be present depending on the data source
        """
        ...


class ExpertReplayBuffer(Sized):
    """Unified interface for expert demonstration replay buffers.

    This class wraps various expert data sources and provides a consistent
    interface for sampling expert demonstrations. It supports:

    1. ImitationLearningTools ExpertReplayManager (primary)
    2. TorchRL TensorDictReplayBuffer
    3. Any object implementing the ExpertBufferProtocol

    Example:
        >>> # With ImitationLearningTools
        >>> from iltools_datasets.replay_manager import ExpertReplayManager, ExpertReplaySpec
        >>> spec = ExpertReplaySpec(...)
        >>> manager = ExpertReplayManager(spec)
        >>> expert_buffer = ExpertReplayBuffer(manager)
        >>> batch = expert_buffer.sample()

        >>> # With TorchRL buffer
        >>> buffer = TensorDictReplayBuffer(...)
        >>> expert_buffer = ExpertReplayBuffer(buffer)
        >>> batch = expert_buffer.sample()
    """

    def __init__(self, source: Any):
        """Initialize the expert replay buffer wrapper.

        Args:
            source: Expert data source. Can be:
                - ImitationLearningTools ExpertReplayManager (with .buffer attribute)
                - TorchRL TensorDictReplayBuffer
                - Any object with a .sample() method that returns TensorDict

        Raises:
            TypeError: If source doesn't implement the required interface
        """
        self._source = source
        self._buffer: TensorDictReplayBuffer | None = None

        # Try to extract the underlying buffer
        if hasattr(source, "buffer") and isinstance(
            getattr(source, "buffer"), TensorDictReplayBuffer
        ):
            # ImitationLearningTools ExpertReplayManager
            self._buffer = source.buffer
        elif isinstance(source, TensorDictReplayBuffer):
            # Direct TorchRL buffer
            self._buffer = source
        elif hasattr(source, "sample") and callable(getattr(source, "sample")):
            # Generic object with sample() method
            self._buffer = None  # Will use source.sample() directly
        else:
            msg = (
                "Expert source must be either:\n"
                "  1. An object with a 'buffer' attribute (TensorDictReplayBuffer)\n"
                "  2. A TensorDictReplayBuffer instance\n"
                "  3. An object with a 'sample()' method\n"
                f"Got: {type(source)}"
            )
            raise TypeError(msg)

    def sample(self) -> TensorDict:
        """Sample a batch of expert demonstrations.

        Returns:
            TensorDict with expert demonstration data

        Raises:
            RuntimeError: If sampling fails
        """
        try:
            if self._buffer is not None:
                return self._buffer.sample()
            else:
                return self._source.sample()
        except Exception as e:
            msg = f"Failed to sample from expert buffer: {e}"
            raise RuntimeError(msg) from e

    @property
    def buffer(self) -> TensorDictReplayBuffer | None:
        """Access the underlying TensorDictReplayBuffer if available.

        Returns:
            The underlying buffer, or None if not applicable
        """
        return self._buffer

    @property
    def source(self) -> Any:
        """Access the original source object.

        Returns:
            The original source object passed to __init__
        """
        return self._source

    def __len__(self) -> int:
        """Return the number of transitions in the expert buffer.

        Returns:
            Number of transitions available

        Raises:
            RuntimeError: If buffer size cannot be determined
        """
        if self._buffer is not None:
            return len(self._buffer)
        elif hasattr(self._source, "__len__"):
            return len(self._source)
        else:
            msg = "Expert buffer does not support len() operation"
            raise RuntimeError(msg)

    def __repr__(self) -> str:
        """String representation of the expert buffer."""
        source_type = type(self._source).__name__
        has_buffer = self._buffer is not None
        return f"ExpertReplayBuffer(source_type={source_type}, has_buffer={has_buffer})"
