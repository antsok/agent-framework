# Copyright (c) Microsoft. All rights reserved.

"""Compaction and prompt-cache interaction benchmark for Agent Framework.

Provider prompt caches match on exact prefixes, and every compaction strategy works by
excluding or rewriting messages inside an existing history. Compaction therefore breaks
the cached prefix by construction, and the question this package answers empirically is
what that costs: how much of each prompt stays reusable under a given strategy, how much
of that reusable prefix a given provider actually serves from cache, and whether
compacting earlier saves more prompt tokens than it loses in cache reads.

Measurements come from replaying deterministic scripted transcripts, so every provider and
every strategy sees a byte-identical conversation.
"""

import importlib.metadata

from ._advise_cli import main as advise_main
from ._advisor import ModelPricing, StrategyCost, Verdict, advise, cost_of, fetch_openrouter_pricing
from ._cli import build_parser, main, run_benchmark
from ._metrics import common_message_prefix, percentile, serialize_message, summarize_cell, token_counts
from ._providers import (
    PROVIDER_SPECS,
    ProviderRuntime,
    ProviderSpec,
    build_provider,
    parse_provider_selector,
    prompt_cache_key,
    prompt_cache_key_options,
    provider_names,
)
from ._recall import (
    Contradiction,
    FactOutcome,
    PlantedFact,
    RecallScenario,
    RecallScore,
    build_recall_scenario,
    score_answer,
)
from ._recall_cli import main as recall_main
from ._report import render_summary_table, write_records_jsonl, write_summary_csv
from ._runner import CallOutcome, ProviderCaller, TurnCaller, run_cell
from ._strategies import (
    STRATEGY_BUILDERS,
    StrategyOptions,
    build_strategy,
    resolve_context_window,
    strategy_names,
)
from ._summary import JointOutcome, JointVerdict, recommend, relative_correctness
from ._summary_cli import main as summary_main
from ._transcripts import DEFAULT_SYSTEM_TOKENS, TRANSCRIPT_PRESETS, TranscriptPreset, build_preset, build_transcript
from ._types import CellKey, CellSummary, Transcript, TranscriptTurn, TurnRecord

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    "DEFAULT_SYSTEM_TOKENS",
    "PROVIDER_SPECS",
    "STRATEGY_BUILDERS",
    "TRANSCRIPT_PRESETS",
    "CallOutcome",
    "CellKey",
    "CellSummary",
    "Contradiction",
    "FactOutcome",
    "JointOutcome",
    "JointVerdict",
    "ModelPricing",
    "PlantedFact",
    "ProviderCaller",
    "ProviderRuntime",
    "ProviderSpec",
    "RecallScenario",
    "RecallScore",
    "StrategyCost",
    "StrategyOptions",
    "Transcript",
    "TranscriptPreset",
    "TranscriptTurn",
    "TurnCaller",
    "TurnRecord",
    "Verdict",
    "__version__",
    "advise",
    "advise_main",
    "build_parser",
    "build_preset",
    "build_provider",
    "build_recall_scenario",
    "build_strategy",
    "build_transcript",
    "common_message_prefix",
    "cost_of",
    "fetch_openrouter_pricing",
    "main",
    "parse_provider_selector",
    "percentile",
    "prompt_cache_key",
    "prompt_cache_key_options",
    "provider_names",
    "recall_main",
    "recommend",
    "relative_correctness",
    "render_summary_table",
    "resolve_context_window",
    "run_benchmark",
    "run_cell",
    "score_answer",
    "serialize_message",
    "strategy_names",
    "summarize_cell",
    "summary_main",
    "token_counts",
    "write_records_jsonl",
    "write_summary_csv",
]
