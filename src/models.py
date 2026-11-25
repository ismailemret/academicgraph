from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Academician:
    """Represents a CS/CENG faculty member and their normalized research fields."""

    acad_id: str
    name: str
    title: str
    university: str
    department: str
    fields: tuple[str, ...] = field(default_factory=tuple)

    def __init__(
        self,
        acad_id: str,
        name: str,
        title: str,
        university: str,
        department: str,
        fields: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> None:
        if not acad_id or not acad_id.strip():
            raise ValueError("Academician 'acad_id' cannot be empty.")
        if not name or not name.strip():
            raise ValueError(f"Academician '{acad_id}' must have a non-empty name.")

        self.acad_id = acad_id.strip()
        self.name = name.strip()
        self.title = (title or "").strip()
        self.university = (university or "Unknown").strip()
        self.department = (department or "Computer Science").strip()

        seen: set[str] = set()
        cleaned_fields: list[str] = []
        for raw_field in fields or []:
            normalized = raw_field.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                cleaned_fields.append(normalized)

        self.fields: tuple[str, ...] = tuple(cleaned_fields)
        self._field_set: frozenset[str] = frozenset(cleaned_fields)

    @property
    def field_set(self) -> frozenset[str]:
        """O(1) lookup set of research fields for Jaccard computations."""
        return self._field_set

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Academician:
        return cls(
            acad_id=str(data["id"]),
            name=str(data["name"]),
            title=str(data.get("title", "")),
            university=str(data.get("university", "Unknown")),
            department=str(data.get("department", "Computer Science")),
            fields=data.get("fields", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.acad_id,
            "name": self.name,
            "title": self.title,
            "university": self.university,
            "department": self.department,
            "fields": list(self.fields),
        }

    def __hash__(self) -> int:
        return hash(self.acad_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Academician):
            return NotImplemented
        return self.acad_id == other.acad_id


@dataclass
class University:
    """Aggregates academicians belonging to a single institution."""

    name: str
    academicians: list[Academician] = field(default_factory=list)

    def add_academician(self, academician: Academician) -> None:
        if academician not in self.academicians:
            self.academicians.append(academician)

    @property
    def academician_count(self) -> int:
        return len(self.academicians)

    def field_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for acad in self.academicians:
            for fld in acad.fields:
                counts[fld] = counts.get(fld, 0) + 1
        return counts

    def top_fields(self, top_n: int = 5) -> list[tuple[str, int]]:
        ranked = sorted(self.field_counts().items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_n]


@dataclass
class Field:
    """Represents a research domain and the academicians active in it."""

    name: str
    academicians: list[Academician] = field(default_factory=list)

    def add_academician(self, academician: Academician) -> None:
        if academician not in self.academicians:
            self.academicians.append(academician)

    @property
    def academician_count(self) -> int:
        return len(self.academicians)


class FieldHierarchy:
    """Directed acyclic taxonomy representing parent-child relationships between research fields."""

    def __init__(self) -> None:
        self._parents: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}
        self._canonical: dict[str, str] = {}

    def _register(self, name: str) -> str:
        cleaned = name.strip()
        key = cleaned.lower()
        if key not in self._canonical:
            self._canonical[key] = cleaned
            self._parents[cleaned] = set()
            self._children[cleaned] = set()
        return self._canonical[key]

    def add_relation(self, parent: str, child: str) -> None:
        if not parent or not child:
            return
        p_can = self._register(parent)
        c_can = self._register(child)
        if p_can != c_can:
            self._children[p_can].add(c_can)
            self._parents[c_can].add(p_can)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[list[str]]) -> FieldHierarchy:
        hierarchy = cls()
        if isinstance(data, list):
            for pair in data:
                if len(pair) == 2:
                    hierarchy.add_relation(str(pair[0]), str(pair[1]))
            return hierarchy

        def _walk(parent_name: str, subtree: Any) -> None:
            hierarchy._register(parent_name)
            if isinstance(subtree, dict):
                for child_name, grand_children in subtree.items():
                    hierarchy.add_relation(parent_name, str(child_name))
                    _walk(str(child_name), grand_children)
            elif isinstance(subtree, list):
                for child_name in subtree:
                    hierarchy.add_relation(parent_name, str(child_name))

        for root_name, children in data.items():
            _walk(str(root_name), children)
        return hierarchy

    @classmethod
    def from_json(cls, path: str | Path) -> FieldHierarchy:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def ancestors(self, field_name: str) -> dict[str, int]:
        canonical = self._canonical.get(field_name.strip().lower())
        if not canonical:
            return {}
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(canonical, 0)])
        while queue:
            current, dist = queue.popleft()
            for parent in self._parents.get(current, ()):
                if parent not in distances:
                    distances[parent] = dist + 1
                    queue.append((parent, dist + 1))
        return distances

    def descendants(self, field_name: str) -> dict[str, int]:
        canonical = self._canonical.get(field_name.strip().lower())
        if not canonical:
            return {}
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(canonical, 0)])
        while queue:
            current, dist = queue.popleft()
            for child in self._children.get(current, ()):
                if child not in distances:
                    distances[child] = dist + 1
                    queue.append((child, dist + 1))
        return distances

    def relation_weight(
        self,
        target_field: str,
        candidate_field: str,
        decay: float = 0.5,
    ) -> float:
        """Returns 1.0 for exact match, decay^distance for ancestor/descendant, else 0.0."""
        t_key = target_field.strip().lower()
        c_key = candidate_field.strip().lower()
        if not t_key or not c_key:
            return 0.0
        if t_key == c_key:
            return 1.0

        t_can = self._canonical.get(t_key)
        c_can = self._canonical.get(c_key)
        if not t_can or not c_can:
            return 0.0

        desc = self.descendants(t_can)
        if c_can in desc:
            return decay ** desc[c_can]

        anc = self.ancestors(t_can)
        if c_can in anc:
            return decay ** anc[c_can]

        return 0.0