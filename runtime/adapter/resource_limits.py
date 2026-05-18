from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class ResourceLimits:
    timeout_seconds: float = 120.0
    max_retries: int = 1
    circuit_breaker_failures: int = 3


class CircuitBreaker:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.failures = 0
        self.open = False

    async def call(self, operation: Awaitable[T]) -> T:
        if self.open:
            raise RuntimeError("circuit breaker open")
        try:
            result = await operation
        except Exception:
            self.failures += 1
            if self.failures >= self.threshold:
                self.open = True
            raise
        self.failures = 0
        return result


async def run_with_limits(operation_factory: Callable[[], Awaitable[T]], limits: ResourceLimits) -> T:
    breaker = CircuitBreaker(limits.circuit_breaker_failures)
    last_error: Exception | None = None
    for _ in range(limits.max_retries + 1):
        try:
            return await asyncio.wait_for(breaker.call(operation_factory()), timeout=limits.timeout_seconds)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error))
