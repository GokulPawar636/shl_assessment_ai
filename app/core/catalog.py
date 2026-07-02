"""
Loads and normalizes the SHL product catalog.
Single source of truth every tool (Metadata, FAISS, Compare) reads from.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "shl_product_catalog.json"

# The canonical "keys" (assessment category) taxonomy observed in the catalog.
# Kept here so tools can validate/normalize LLM-provided filter values.
KNOWN_KEYS = [
    "Ability & Aptitude",
    "Assessment Exercises",
    "Biodata & Situational Judgment",
    "Competencies",
    "Development & 360",
    "Knowledge & Skills",
    "Personality & Behavior",
    "Simulations",
]

# Short-code mapping used in the response tables in the example transcripts
# (A, P, K, S, B, C, D). Useful for rendering, not for filtering.
KEY_SHORT_CODE = {
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

KNOWN_JOB_LEVELS = [
    "Director",
    "Entry-Level",
    "Executive",
    "Front Line Manager",
    "General Population",
    "Graduate",
    "Manager",
    "Mid-Professional",
    "Professional Individual Contributor",
    "Supervisor",
]


@dataclass
class Product:
    entity_id: str
    name: str
    link: str
    job_levels: list[str]
    languages: list[str]
    duration: str
    duration_minutes: int | None
    adaptive: bool
    remote: bool
    description: str
    keys: list[str]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def key_codes(self) -> str:
        return ",".join(KEY_SHORT_CODE.get(k, "?") for k in self.keys)

    def searchable_text(self) -> str:
        """Text blob used for embedding / semantic search."""
        return (
            f"{self.name}. "
            f"Categories: {', '.join(self.keys)}. "
            f"Job levels: {', '.join(self.job_levels)}. "
            f"{self.description}"
        )

    def to_row(self, index: int) -> dict[str, Any]:
        langs = self.languages
        lang_str = ", ".join(langs[:4])
        if len(langs) > 4:
            lang_str += f" (+{len(langs) - 4} more)"
        elif not langs:
            lang_str = "—"
        return {
            "#": index,
            "name": self.name,
            "test_type": self.key_codes(),
            "keys": ", ".join(self.keys),
            "duration": self.duration or "—",
            "languages": lang_str,
            "url": self.link,
        }


_DURATION_RE = re.compile(r"(\d+)")


def _parse_duration_minutes(duration_str: str) -> int | None:
    if not duration_str:
        return None
    m = _DURATION_RE.search(duration_str)
    return int(m.group(1)) if m else None


def _load_raw(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    # The scraped catalog contains a few literal control characters inside
    # description strings. strict=False tells the decoder to tolerate them
    # instead of raising JSONDecodeError.
    return json.loads(text, strict=False)


class Catalog:
    """In-memory catalog of SHL products with lookup helpers."""

    def __init__(self, path: Path | str = DEFAULT_CATALOG_PATH):
        self.path = Path(path)
        self.products: list[Product] = []
        self._by_id: dict[str, Product] = {}
        self._by_name_lower: dict[str, Product] = {}
        self._load()

    def _load(self) -> None:
        raw_items = _load_raw(self.path)
        for item in raw_items:
            if item.get("status") != "ok":
                continue
            duration_raw = item.get("duration", "") or ""
            product = Product(
                entity_id=str(item.get("entity_id", "")),
                name=item.get("name", "").strip(),
                link=item.get("link", ""),
                job_levels=list(item.get("job_levels", []) or []),
                languages=list(item.get("languages", []) or []),
                duration=duration_raw,
                duration_minutes=_parse_duration_minutes(duration_raw),
                adaptive=(item.get("adaptive") == "yes"),
                remote=(item.get("remote") == "yes"),
                description=(item.get("description", "") or "").strip(),
                keys=list(item.get("keys", []) or []),
                raw=item,
            )
            self.products.append(product)
            self._by_id[product.entity_id] = product
            self._by_name_lower[product.name.lower()] = product

    def __len__(self) -> int:
        return len(self.products)

    def get_by_id(self, entity_id: str) -> Product | None:
        return self._by_id.get(str(entity_id))

    def get_by_name(self, name: str) -> Product | None:
        """Exact (case-insensitive) name match, falling back to substring match."""
        exact = self._by_name_lower.get(name.lower())
        if exact:
            return exact
        name_lower = name.lower()
        candidates = [p for p in self.products if name_lower in p.name.lower()]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def find_by_names(self, names: list[str]) -> list[Product]:
        found = []
        for n in names:
            p = self.get_by_name(n)
            if p:
                found.append(p)
        return found
