"""
Recommendation Agent.

Takes raw ToolResult(s) from the Tool Decision Engine plus the Planner's
decision (default additions, exclusions from memory) and produces a single
ordered shortlist of Products — the thing that gets rendered as the
markdown table in every example transcript.

Merge policy:
  - metadata_filter and semantic_search results are unioned, de-duplicated
    by entity_id, preserving first-seen order (metadata results first,
    since they represent hard constraints the user stated).
  - semantic_search scores (if present) are used to order results within
    the semantic portion.
  - default_additions (e.g. OPQ32r) are appended at the end if not already
    present and not user-excluded.
  - anything in state.excluded_ids is filtered out unconditionally, even if
    a tool call re-surfaces it.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.catalog import Catalog, Product
from app.core.memory import SessionState
from app.core.tool_engine import ToolResult


@dataclass
class Recommendation:
    products: list[Product]

    def to_table_rows(self) -> list[dict]:
        return [p.to_row(i + 1) for i, p in enumerate(self.products)]

    def entity_ids(self) -> list[str]:
        return [p.entity_id for p in self.products]


class RecommendationAgent:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def build(
        self,
        tool_results: list[ToolResult],
        default_addition_names: list[str],
        state: SessionState,
        max_items: int = 8,
    ) -> Recommendation:
        seen: set[str] = set()
        ordered: list[Product] = []

        # Hard-constraint (metadata) results first.
        for tr in tool_results:
            if tr.tool != "metadata_filter" or tr.error:
                continue
            for p in tr.products:
                if p.entity_id in state.excluded_ids or p.entity_id in seen:
                    continue
                ordered.append(p)
                seen.add(p.entity_id)

        # Then semantic results, best score first.
        for tr in tool_results:
            if tr.tool != "semantic_search" or tr.error:
                continue
            ranked = sorted(tr.products, key=lambda p: tr.scores.get(p.entity_id, 0.0), reverse=True)
            for p in ranked:
                if p.entity_id in state.excluded_ids or p.entity_id in seen:
                    continue
                ordered.append(p)
                seen.add(p.entity_id)

        # Default additions (e.g. OPQ32r) appended last, unless excluded.
        for name in default_addition_names:
            p = self.catalog.get_by_name(name)
            if p and p.entity_id not in state.excluded_ids and p.entity_id not in seen:
                ordered.append(p)
                seen.add(p.entity_id)

        return Recommendation(products=ordered[:max_items])

    def carry_forward(self, state: SessionState) -> Recommendation:
        """Re-render the last shown set (minus any newly excluded ids) — used on finalize."""
        products = []
        for eid in state.last_recommendation_ids:
            if eid in state.excluded_ids:
                continue
            p = self.catalog.get_by_id(eid)
            if p:
                products.append(p)
        return Recommendation(products=products)
