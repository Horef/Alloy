"""Offline A/B probe for the opt-in theme-aware topic layer.

Builds the corpus knowledge graph twice from the SAME cached node signals -- once in the default
entity clustering mode and once in the theme-first mode -- and prints both topic maps side by side.
Theme mode re-extracts node signals once (the theme_signature changes the per-document cache key),
which is a one-time cost; afterwards both graphs are cheap to rebuild.

Usage:
    python docs/theme_ab_probe.py --config hova/hova_config.toml --documents hova/hova_knowledge_base \
        --cache hova/.chatbot_eval_cache --look-for שכר
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from chatbot_eval.cache import CorpusAnalysisCache
from chatbot_eval.cli import _build_graph_bundle
from chatbot_eval.config import load_settings
from chatbot_eval.llm import GeminiStructuredLLM


def _print_topic_map(title: str, bundle) -> None:
    print(f"\n=== {title} : {len(bundle.topics)} topics ===")
    for topic in sorted(bundle.topics, key=lambda t: (-t.importance, t.name)):
        print(f"  [{topic.importance}] {topic.name}  ({len(topic.node_ids)} chunks)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--look-for", action="append", default=[],
                        help="Substring(s) to check for in topic names (e.g. a theme the team said was missing)")
    args = parser.parse_args()

    settings = load_settings(args.config)
    llm = GeminiStructuredLLM(
        settings.api_key, settings.max_retries, transport=settings.gemini_transport,
        apigee_api_key=settings.apigee_api_key, apigee_base_url=settings.apigee_base_url,
        request_timeout_seconds=settings.gemini_request_timeout_seconds,
    )

    def build(mode: str):
        tuned = dataclasses.replace(settings, topic_mode=mode)
        cache = CorpusAnalysisCache(args.cache, enabled=True)
        chunks, _ = cache.load_chunks(
            args.documents, settings.chunk_chars, settings.chunk_overlap_chars, progress_enabled=False,
        )
        return _build_graph_bundle(chunks, cache, tuned, llm, progress_enabled=False,
                                   max_concurrency=args.max_concurrency)

    entity = build("entity")
    _print_topic_map("ENTITY MODE (default)", entity)
    theme = build("theme")
    _print_topic_map("THEME MODE (opt-in)", theme)

    for needle in args.look_for:
        e_hit = [t.name for t in entity.topics if needle in t.name]
        t_hit = [t.name for t in theme.topics if needle in t.name]
        print(f"\n--- '{needle}' ---")
        print(f"  entity mode topic names containing it: {e_hit or 'NONE'}")
        print(f"  theme  mode topic names containing it: {t_hit or 'NONE'}")

    # Hairball guard: report the largest topic in each mode.
    e_max = max((len(t.node_ids) for t in entity.topics), default=0)
    t_max = max((len(t.node_ids) for t in theme.topics), default=0)
    print(f"\nlargest topic size -- entity: {e_max} chunks | theme: {t_max} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
