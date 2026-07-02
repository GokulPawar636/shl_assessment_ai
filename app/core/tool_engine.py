"""
Tool Decision Engine.

Sits between the LLM Planner and the three tools. The planner emits one or
more tool *requests* (structured dicts); this engine validates them,
executes each tool (potentially in parallel — they're independent and
read-only), and normalizes results into a common shape the Recommendation
Agent can merge regardless of which tool(s) produced them.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from app.core.catalog import Catalog, Product
from app.tools.compare_tool import CompareTool
from app.tools.faiss_tool import FaissTool
from app.tools.metadata_tool import MetadataFilter, MetadataTool


@dataclass
class ToolRequest:
    tool: str  # "metadata_filter" | "semantic_search" | "compare_products"
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool: str
    products: list[Product] = field(default_factory=list)
    # score per entity_id, if the tool produced relevance scores (FAISS does)
    scores: dict[str, float] = field(default_factory=dict)
    comparison: Any = None  # CompareTool.run() output, when tool == compare_products
    error: str | None = None


class ToolDecisionEngine:
    def __init__(self, catalog: Catalog, faiss_tool: FaissTool | None = None):
        self.catalog = catalog
        self.metadata_tool = MetadataTool(catalog)
        self.faiss_tool = faiss_tool or FaissTool(catalog)
        self.compare_tool = CompareTool(catalog)

    def _run_one(self, request: ToolRequest, exclude_ids: set[str]) -> ToolResult:
        try:
            if request.tool == "metadata_filter":
                args = dict(request.args)
                merged_exclude = set(args.get("exclude_ids") or []) | exclude_ids
                args["exclude_ids"] = merged_exclude
                filt = MetadataTool.from_dict(args)
                products = self.metadata_tool.run(filt, limit=args.get("limit", 25))
                return ToolResult(tool=request.tool, products=products)

            if request.tool == "semantic_search":
                query = request.args.get("query", "")
                top_k = request.args.get("top_k", 10)
                pairs = self.faiss_tool.run(query, top_k=top_k, exclude_ids=exclude_ids)
                products = [p for p, _ in pairs]
                scores = {p.entity_id: s for p, s in pairs}
                return ToolResult(tool=request.tool, products=products, scores=scores)

            if request.tool == "compare_products":
                names = request.args.get("product_names", [])
                comparison = self.compare_tool.run(names)
                return ToolResult(tool=request.tool, products=comparison.products, comparison=comparison)

            return ToolResult(tool=request.tool, error=f"Unknown tool: {request.tool}")
        except Exception as e:  # tools must never crash the conversation turn
            return ToolResult(tool=request.tool, error=str(e))

    def run(self, requests: list[ToolRequest], exclude_ids: set[str] | None = None) -> list[ToolResult]:
        exclude_ids = exclude_ids or set()
        if not requests:
            return []
        if len(requests) == 1:
            return [self._run_one(requests[0], exclude_ids)]
        # Independent, read-only calls -> safe to fan out concurrently.
        with ThreadPoolExecutor(max_workers=min(4, len(requests))) as ex:
            return list(ex.map(lambda r: self._run_one(r, exclude_ids), requests))
