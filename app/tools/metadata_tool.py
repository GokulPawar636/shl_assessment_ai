"""
Metadata Tool: structured/exact-match filtering over the catalog.

Handles queries the LLM planner can express as hard constraints, e.g.
"job level = Graduate", "language includes Latin American Spanish",
"keys includes Personality & Behavior", "duration <= 20 minutes",
"adaptive = true". This is the right tool whenever the user's ask maps
cleanly onto catalog fields rather than needing free-text similarity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.catalog import Catalog, Product


@dataclass
class MetadataFilter:
    job_levels: list[str] | None = None          # OR within field
    languages: list[str] | None = None            # OR within field
    keys: list[str] | None = None                 # OR within field (category)
    keys_all: list[str] | None = None              # AND — every listed key must be present
    max_duration_minutes: int | None = None
    min_duration_minutes: int | None = None
    adaptive: bool | None = None
    name_contains: str | None = None
    exclude_ids: set[str] | None = None


class MetadataTool:
    name = "metadata_filter"
    description = (
        "Filter the SHL catalog by exact structured fields: job level, language, "
        "category (keys), duration bounds, and adaptive flag. Use this when the "
        "request specifies concrete constraints rather than a fuzzy topic."
    )

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def run(self, filt: MetadataFilter, limit: int = 25) -> list[Product]:
        # name_contains is meant to pin ONE specific product (planner names
        # it precisely, e.g. "SQL (New)"). Pure substring matching over-fires
        # on names like "Oracle PL/SQL (New)" or "Automata - SQL (New)".
        # Resolution order: exact match wins outright; otherwise fall back to
        # substring matching only among names that START WITH the query,
        # which is far less prone to false positives from compound names.
        if filt.name_contains:
            exact = self.catalog.get_by_name(filt.name_contains)
            if exact and not (filt.exclude_ids and exact.entity_id in filt.exclude_ids):
                return [exact]
            query_lower = filt.name_contains.lower()
            prefix_matches = [
                p for p in self.catalog.products
                if p.name.lower().startswith(query_lower)
                and not (filt.exclude_ids and p.entity_id in filt.exclude_ids)
            ]
            if prefix_matches:
                return self._apply_remaining_filters(prefix_matches, filt)[:limit]
            # last resort: substring, still filtered by the rest of the criteria
            substring_matches = [
                p for p in self.catalog.products
                if query_lower in p.name.lower()
                and not (filt.exclude_ids and p.entity_id in filt.exclude_ids)
            ]
            return self._apply_remaining_filters(substring_matches, filt)[:limit]

        return self._apply_remaining_filters(
            [p for p in self.catalog.products if not (filt.exclude_ids and p.entity_id in filt.exclude_ids)],
            filt,
        )[:limit]

    @staticmethod
    def _apply_remaining_filters(candidates: list[Product], filt: MetadataFilter) -> list[Product]:
        """Applies every filter field EXCEPT name_contains (already resolved
        by the caller) to a candidate list."""
        results = []
        for p in candidates:
            if filt.job_levels and not (set(filt.job_levels) & set(p.job_levels)):
                continue
            if filt.languages and not (set(filt.languages) & set(p.languages)):
                continue
            if filt.keys and not (set(filt.keys) & set(p.keys)):
                continue
            if filt.keys_all and not set(filt.keys_all).issubset(set(p.keys)):
                continue
            if filt.max_duration_minutes is not None:
                if p.duration_minutes is not None and p.duration_minutes > filt.max_duration_minutes:
                    continue
            if filt.min_duration_minutes is not None:
                if p.duration_minutes is not None and p.duration_minutes < filt.min_duration_minutes:
                    continue
            if filt.adaptive is not None and p.adaptive != filt.adaptive:
                continue
            results.append(p)
        return results

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MetadataFilter":
        return MetadataFilter(
            job_levels=d.get("job_levels"),
            languages=d.get("languages"),
            keys=d.get("keys"),
            keys_all=d.get("keys_all"),
            max_duration_minutes=d.get("max_duration_minutes"),
            min_duration_minutes=d.get("min_duration_minutes"),
            adaptive=d.get("adaptive"),
            name_contains=d.get("name_contains"),
            exclude_ids=set(d["exclude_ids"]) if d.get("exclude_ids") else None,
        )
