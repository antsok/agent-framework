# Copyright (c) Microsoft. All rights reserved.

"""The replay loop.

One cell is one scripted transcript replayed against one provider under one compaction
strategy. Each turn appends the scripted request messages to history, runs compaction over
that history exactly as ``CompactionProvider.before_run`` would, sends the resulting
projection to the provider, and then appends the scripted reply. The model's real answer
is discarded, which is what keeps every provider and every strategy on a byte-identical
conversation.

Cells are run sequentially by default. Provider caches are shared, rate-limited, and
eviction-prone, so overlapping cells would contaminate each other's measurements.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from agent_framework import Message, apply_compaction

from ._metrics import common_message_prefix, serialize_message
from ._types import CellKey, TurnRecord

if TYPE_CHECKING:
    from agent_framework import CompactionStrategy, TokenizerProtocol

    from ._providers import ProviderRuntime
    from ._types import Transcript

__all__ = [
    "CallOutcome",
    "ProviderCaller",
    "TurnCaller",
    "is_rate_limited",
    "retry_after_seconds",
    "run_cell",
    "unsupported_option",
]

logger = logging.getLogger("agent_framework_lab_cachebench")

_RATE_LIMIT_MARKERS: Final[tuple[str, ...]] = ("rate limit", "rate_limit", "too many requests", "throttl")

# The bare status number, matched on a digit boundary rather than as a substring. "429" also
# sits inside plenty of token counts -- a prompt-too-large error reporting 274,293 tokens
# contains it -- and a deterministic 400 retried as if it were throttling wastes the attempt
# budget and delays the honest failure by minutes.
_STATUS_429: Final[re.Pattern[str]] = re.compile(r"(?<!\d)429(?!\d)")

# Reasoning models reject sampling parameters outright. The wording differs by provider,
# and some gateways silently strip the field instead of failing, so the same model can 400
# on one route and succeed on another.
_UNSUPPORTED_PARAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[Uu]nsupported parameter: '([a-z_]+)'"),
    re.compile(r"'([a-z_]+)' is not supported with this model"),
    re.compile(r"[Uu]nrecognized request argument supplied: ([a-z_]+)"),
    # Some providers reject forcing without naming the field: Z.AI answers a pinned
    # tool_choice with "Tool choice must be auto". Map it back to the option it refers to
    # so the same drop-and-retry path applies.
    re.compile(r"[Tt]ool choice must be auto()"),
)


def unsupported_option(error: BaseException) -> str | None:
    """Return the request option a provider rejected outright, if it named one.

    The framework wraps provider errors in an exception whose args are a tuple, so
    ``str()`` renders the payload through ``repr`` and the quotes around the parameter
    name arrive backslash-escaped. Normalizing them away first keeps the patterns readable
    and lets the same pattern match whether the message was repr'd or not.
    """
    text = str(error).replace("\\'", "'").replace('\\"', '"')
    for pattern in _UNSUPPORTED_PARAM_PATTERNS:
        if match := pattern.search(text):
            return match.group(1) or "tool_choice"
    return None


def _error_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and everything it was raised from, outermost first.

    The framework wraps a provider's error in ``ChatClientException``, so the object that
    knows which HTTP status arrived is never the object a caller catches. Guarded against a
    cycle, because ``__context__`` is set implicitly and two exceptions raised inside each
    other's handlers point at one another.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(error: BaseException) -> int | None:
    """Return the HTTP status an exception carries, wherever it happens to carry it.

    Providers disagree on the attribute name and on whether it sits on the error or on a
    response hanging off it, and one of them reports it as a string.
    """
    for holder in (error, getattr(error, "response", None)):
        if holder is None:
            continue
        for name in ("status_code", "status", "http_status", "code"):
            value = getattr(holder, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            # ``code`` is a string on the OpenAI SDK and carries ``rate_limit_exceeded``
            # rather than a number, so only a numeric one is a status.
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def is_rate_limited(error: BaseException) -> bool:
    """Return whether a provider refused this call for rate reasons rather than for content.

    Structure before wording. A 429 reaches us as ``ChatClientException`` wrapping the
    provider SDK's own error, and every provider wraps it differently, so matching on the
    exception class would work for exactly one of them. A status of 429 anywhere in the chain
    is the one signal that does not depend on phrasing; the markers are the fallback for a
    wrapper that rendered its cause to a string and dropped the object it came from.

    Args:
        error: The exception a call raised.

    Returns:
        True when waiting and re-sending is the right response.
    """
    if any(_status_code(exc) == 429 for exc in _error_chain(error)):
        return True
    text = " ".join(f"{type(exc).__name__} {exc}" for exc in _error_chain(error)).lower()
    return bool(_STATUS_429.search(text)) or any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _header(headers: Any, name: str) -> str | None:
    """Return one header by name, case-insensitively, from whatever shape it arrived in.

    Scanned rather than looked up. ``httpx.Headers`` is already case-insensitive and a plain
    dict is not, and which of the two arrives depends on how far up the chain the object was
    found -- so the case handling has to live here rather than be assumed of the caller.
    """
    try:
        pairs: Iterable[tuple[Any, Any]] = headers.items()
    except (AttributeError, TypeError):
        # Not a mapping at all: this walks whatever objects the chain happens to hang off an
        # exception, so "no headers here" is the ordinary case rather than an error.
        return None
    return next((str(value) for key, value in pairs if str(key).lower() == name), None)


def _seconds(raw: Any) -> float | None:
    """Return a non-negative number of seconds, or None when the value is not one.

    ``Retry-After`` is allowed to be an HTTP date rather than a count, and a date is not worth
    parsing here: the exponential schedule is a safe answer for one, whereas a misparsed date
    is a wait of hours or of nothing.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def retry_after_seconds(error: BaseException) -> float | None:
    """Return the wait the provider asked for, if anything in the chain named one.

    Preferred over any schedule invented here, because the provider knows when its own window
    refills. Guessing shorter spends an attempt against a limit that has not moved.

    Args:
        error: The exception a call raised.

    Returns:
        Seconds to wait, or None when no delay was named.
    """
    for exc in _error_chain(error):
        for holder in (exc, getattr(exc, "response", None)):
            if holder is None:
                continue
            if (named := _seconds(getattr(holder, "retry_after", None))) is not None:
                return named
            headers = getattr(holder, "headers", None)
            for name, scale in (("retry-after", 1.0), ("retry-after-ms", 0.001)):
                if (value := _seconds(_header(headers, name))) is not None:
                    return value * scale
    return None


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """What one model call reported back."""

    latency_ms: float
    input_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    text: str | None = None
    """The model's answer. Discarded by the cost benchmark, which replays scripted replies
    to keep prompts identical, but required by the recall probe, which scores what the model
    actually said."""


class TurnCaller(Protocol):
    """Issues one model call and reports its usage.

    Implemented by :class:`ProviderCaller` for live runs and by a stub for dry runs, which
    lets the whole replay loop be exercised without spending anything.
    """

    async def __call__(self, messages: Sequence[Message]) -> CallOutcome:
        """Send ``messages`` and return the resulting usage measurements."""
        ...


def _merge_options(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge per-cell options over per-provider ones, merging ``extra_body`` key-wise.

    ``extra_body`` carries several independent concerns at once — OpenRouter provider
    pinning, usage accounting, and the cell's cache key — so replacing it wholesale would
    silently drop whichever the other layer set.

    Args:
        base: The provider's standing options.
        overlay: Per-cell additions.

    Returns:
        A new merged options dict.
    """
    merged = dict(base)
    for key, value in overlay.items():
        if key == "extra_body" and isinstance(value, Mapping):
            merged["extra_body"] = {**(merged.get("extra_body") or {}), **value}
        else:
            merged[key] = value
    return merged


class ProviderCaller:
    """Calls a real provider and extracts normalized usage from the response.

    The Agent Framework already maps each provider's native usage shape onto
    ``cache_read_input_token_count`` and ``cache_creation_input_token_count``, so no
    provider-specific parsing is needed here. Fields a provider omits stay ``None`` and
    are reported as unavailable rather than as zero.
    """

    def __init__(
        self,
        runtime: ProviderRuntime,
        *,
        extra_options: Mapping[str, Any] | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        request_timeout: float | None = None,
    ) -> None:
        """Create a caller.

        Args:
            runtime: The provider client, model, and per-request options. The client is
                shared across cells so its connection pool is reused.

        Keyword Args:
            extra_options: Per-cell options merged over the runtime's, such as the cell's
                ``prompt_cache_key``. ``extra_body`` is merged key-wise rather than
                replaced, so a cell key cannot drop a provider's routing configuration.
            max_retries: Attempts made after throttling before the turn is recorded as an
                error.
            retry_base_delay: Seconds for the first backoff, doubled per retry.
            request_timeout: Seconds a single call may take before it is abandoned and
                recorded as an error. Without a bound, one queue-happy provider can wedge a
                multi-hour sweep indefinitely: the underlying SDK's own timeout stacks with
                its internal retries and with this class's, so a single wedged call can
                absorb hours. ``None`` disables the bound.
        """
        self.runtime = runtime
        # A private copy: the unsupported-parameter retry mutates this, and the runtime's
        # options are shared with every other cell on the same provider.
        self.options = _merge_options(runtime.options, extra_options or {})
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.request_timeout = request_timeout

    async def __call__(self, messages: Sequence[Message]) -> CallOutcome:
        """Send ``messages``, retrying on throttling, and return the usage measurements."""
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                # Options travel as a single ``options`` mapping, not as **kwargs: the
                # layered client signature is
                # ``get_response(messages, *, stream, options, middleware, ...)``
                # and splatting request options into it raises TypeError.
                call = self.runtime.client.get_response(list(messages), options=self.options)
                response = await (
                    asyncio.wait_for(call, timeout=self.request_timeout) if self.request_timeout is not None else call
                )
            except TimeoutError:
                # Abandoned, not retried: a provider that blew the budget once will very
                # likely do it again, and the point of the bound is to cap total run time.
                return CallOutcome(
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    error=f"TimeoutError: exceeded {self.request_timeout}s",
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries and is_rate_limited(exc):
                    await asyncio.sleep(self.retry_base_delay * (2**attempt))
                    continue
                # A rejected sampling parameter is deterministic, not transient: drop the
                # option the provider named and retry immediately. Without this a single
                # reasoning model fails every turn of every one of its cells, and the run
                # returns nothing for it.
                if (option := unsupported_option(exc)) and option in self.options:
                    self.options.pop(option)
                    logger.warning("Provider rejected %r; retrying without it", option)
                    continue
                return CallOutcome(
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            latency_ms = (time.perf_counter() - started) * 1000.0
            usage: dict[str, Any] = dict(response.usage_details or {})
            return CallOutcome(
                latency_ms=latency_ms,
                input_tokens=usage.get("input_token_count"),
                cached_tokens=usage.get("cache_read_input_token_count"),
                cache_write_tokens=usage.get("cache_creation_input_token_count"),
                output_tokens=usage.get("output_token_count"),
                text=response.text,
            )
        return CallOutcome(latency_ms=0.0, error=f"exhausted retries: {last_error}")


async def run_cell(
    *,
    cell: CellKey,
    transcript: Transcript,
    strategy: CompactionStrategy | None,
    tokenizer: TokenizerProtocol,
    caller: TurnCaller,
    on_record: Callable[[TurnRecord], None] | None = None,
    turn_delay: float = 0.0,
) -> list[TurnRecord]:
    """Replay one transcript end to end and return a record per turn.

    Keyword Args:
        cell: Identity recorded on every returned record.
        transcript: The scripted conversation to replay.
        strategy: Compaction strategy applied before each call, or ``None`` for the
            uncompacted baseline.
        tokenizer: Token counter used both by compaction and by the local prefix oracle.
        caller: Issues the actual model call.
        on_record: Optional callback invoked as each turn completes, for streaming output
            to disk so a long run is not lost if it is interrupted.
        turn_delay: Seconds to wait between turns. Useful against strict rate limits.

    Returns:
        One record per turn, in order.
    """
    history: list[Message] = [transcript.system]
    previous_serialized: list[str] = []
    records: list[TurnRecord] = []

    for turn_index, turn in enumerate(transcript.turns, start=1):
        history.extend(turn.request)
        projected = await apply_compaction(history, strategy=strategy, tokenizer=tokenizer)

        serialized = [serialize_message(message) for message in projected]
        token_counts = [tokenizer.count_tokens(text) for text in serialized]
        prefix_messages = common_message_prefix(previous_serialized, serialized)

        outcome = await caller(projected)
        record = TurnRecord(
            cell=cell,
            turn=turn_index,
            history_messages=len(history),
            sent_messages=len(projected),
            sent_tokens_local=sum(token_counts),
            reusable_prefix_tokens_local=sum(token_counts[:prefix_messages]),
            prefix_broken=bool(previous_serialized) and prefix_messages < len(previous_serialized),
            input_tokens=outcome.input_tokens,
            cached_tokens=outcome.cached_tokens,
            cache_write_tokens=outcome.cache_write_tokens,
            output_tokens=outcome.output_tokens,
            latency_ms=outcome.latency_ms,
            error=outcome.error,
        )
        records.append(record)
        if on_record is not None:
            on_record(record)

        history.extend(turn.reply)
        previous_serialized = serialized
        if turn_delay > 0:
            await asyncio.sleep(turn_delay)

    return records
