"""
Compare Tool: structured diff between two or more named products.

Handles turns like "What's the difference between the DSI and Safety &
Dependability 8.0?" or "Is Advanced Java the right pick vs Entry-Level?".
Produces a structured diff the Response Generator can turn into prose;
it does not itself decide which product is "better" (that's the LLM's job,
grounded in this structured comparison rather than free recall).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.catalog import Catalog, Product


@dataclass
class ComparisonField:
    label: str
    values: dict[str, str]  # product name -> field value string


@dataclass
class ComparisonResult:
    products: list[Product]
    fields: list[ComparisonField]
    not_found: list[str]


class CompareTool:
    name = "compare_products"
    description = (
        "Compare two or more named SHL products field-by-field (category, job "
        "levels, languages, duration, adaptive, description) to answer "
        "'what's the difference between X and Y' style questions."
    )

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def run(self, product_names: list[str]) -> ComparisonResult:
        found: list[Product] = []
        not_found: list[str] = []
        for name in product_names:
            p = self.catalog.get_by_name(name)
            if p:
                found.append(p)
            else:
                not_found.append(name)

        def field(label: str, getter) -> ComparisonField:
            return ComparisonField(
                label=label,
                values={p.name: getter(p) for p in found},
            )

        fields = [
            field("Category", lambda p: ", ".join(p.keys) or "—"),
            field("Job levels", lambda p: ", ".join(p.job_levels) or "—"),
            field("Duration", lambda p: p.duration or "—"),
            field("Adaptive", lambda p: "Yes" if p.adaptive else "No"),
            field("Languages", lambda p: f"{len(p.languages)} language(s)"),
            field("Description", lambda p: p.description),
        ]

        return ComparisonResult(products=found, fields=fields, not_found=not_found)
