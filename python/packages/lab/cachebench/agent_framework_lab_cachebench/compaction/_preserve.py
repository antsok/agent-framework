# Copyright (c) Microsoft. All rights reserved.

"""One annotation that says a message must survive compaction intact.

**The failure this exists for was measured, and it is the worst kind: a strategy destroying
its own output.** :class:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy`
asks the model to write everything that matters from the earlier tool results into one tool
result of its own, and then deletes the groups that record replaced. When the record does not
free enough on its own it hands what is left to a fallback strategy, which defaults to
:class:`~._anchored.AnchoredCompactionStrategy` -- and that strategy shortens and sheds tool
results. *The record is a tool result.* Nothing in ``_anchored`` had ever heard of a record,
so it treated the one artefact the whole design exists to produce as ordinary trimmable bulk.
Measured on a live seed: a record holding four lookups' worth of values, thirty-two
identifiers, reached the prompt the questions were answered from carrying two lookups' worth.
16,617 tokens left while only three messages did, which is shortening rather than deletion,
and the row reported no loss of any kind because every counter it had was counting messages.

That is not a bug in one branch. Every deletion the record licences is only safe *because* the
record is there, so trimming the record after the fact discards the sole surviving copy of
everything the strategy already deleted, and does so at the exact moment the conversation is
under most pressure -- which is when the record is worth most.

**Preserved is not excluded, and the difference is the whole design.** ``EXCLUDED_KEY`` in
``agent_framework._compaction`` says "this message is not being sent", so an excluded message
costs nothing and every token count skips it. A preserved message *is* being sent, is counted
in full, and pays for itself like anything else; the flag says only that no strategy may buy
budget by making it smaller. The two are orthogonal, and a strategy consulting the wrong one
would either send a message it meant to drop or price the prompt as though the record were
free.

**A strategy that cannot reach its ceiling must stop, not spin.** Because a preserved message
still counts, protecting it can leave a conversation over budget with nothing left that may be
removed. The contract in that case is to stop and let the caller see the prompt is over the
ceiling -- the same thing the anchored strategy already does when the anchors alone exceed it.
A shed step that kept re-examining a band it is no longer allowed to touch would loop, and a
step that silently gave up while reporting success would hand the provider a prompt it will
reject. Both are worse than an honest overflow.

**Why a module of its own.** ``_toolsummary`` marks the record and ``_anchored`` honours the
mark, and ``_toolsummary`` already imports ``_anchored`` for its default fallback. Putting the
key in either of them would either invert that dependency or create a cycle. The naming and
the annotate/read pattern follow ``EXCLUDED_KEY``, ``EXCLUDE_REASON_KEY`` and ``set_excluded``
deliberately: a reader who knows the framework's convention should not have to learn a second
one to read this.

**The mark does not survive storage, and must be re-applied.** Compaction runs against a
freshly loaded conversation on every turn and ``additional_properties`` set by a previous pass
is not there when the next one starts -- which is why the framework's own exclusion flags are
re-derived each time too. Whoever owns a message that must be protected therefore re-marks it
on every pass, as ``_toolsummary`` does when it observes a record, rather than marking it once
and trusting it to persist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent_framework import Message

__all__ = [
    "PRESERVED_KEY",
    "PRESERVE_REASON_KEY",
    "any_preserved",
    "is_preserved",
    "set_preserved",
]

#: Marks a message no strategy may shorten, drop or shed. Named after ``EXCLUDED_KEY``, and
#: read the same way: absent means false, so an unannotated conversation behaves exactly as it
#: did before this existed.
PRESERVED_KEY: Final[str] = "_preserved"

#: Why a message was preserved, so a caller inspecting a conversation can tell which strategy
#: claimed it. Mirrors ``EXCLUDE_REASON_KEY``.
PRESERVE_REASON_KEY: Final[str] = "_preserve_reason"


def set_preserved(message: Message, *, preserved: bool, reason: str | None = None) -> bool:
    """Mark ``message`` as protected from removal, or release it.

    Args:
        message: The message to annotate, mutated in place.

    Keyword Args:
        preserved: True to protect it, False to release it.
        reason: Recorded alongside the flag when given, so a conversation can be read back and
            the protecting strategy named. Left untouched when None, which keeps an earlier
            reason rather than blanking it.

    Returns:
        True if the flag's value changed, matching ``set_excluded``'s contract so a caller can
        fold it into a "did anything change" tally without a special case.
    """
    changed = bool(message.additional_properties.get(PRESERVED_KEY, False)) != preserved
    if changed:
        message.additional_properties[PRESERVED_KEY] = preserved
    if reason is not None:
        message.additional_properties[PRESERVE_REASON_KEY] = reason
    return changed


def is_preserved(message: Message) -> bool:
    """Return whether ``message`` may not be shortened, dropped or shed.

    Args:
        message: The message to inspect.

    Returns:
        True when the message carries the protection.
    """
    return bool(message.additional_properties.get(PRESERVED_KEY, False))


def any_preserved(messages: Iterable[Message]) -> bool:
    """Return whether any of ``messages`` is protected.

    Group-level removal is all-or-nothing -- a tool call without its result is a malformed
    conversation on most providers -- so a group is protected as soon as one of its members is,
    rather than only when all of them are. That is also the direction this package errs in
    everywhere else: keeping too much costs tokens a reader can see on the bill, and keeping
    too little costs a fact with no trace of where it went.

    Args:
        messages: The messages to inspect, typically one group's span.

    Returns:
        True when at least one is protected.
    """
    return any(is_preserved(message) for message in messages)
