from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class RouteSource:
    planner: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner": self.planner,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteSource":
        return cls(
            planner=data["planner"],
            parameters=dict(data.get("parameters") or {}),
        )


@dataclass(slots=True)
class Route:
    route_id: str
    product: str
    synthesizable: bool
    reactions: list[str] | None = None
    source: RouteSource | None = None
    length: int = field(init=False)

    def __post_init__(self) -> None:
        self.length = len(self.reactions) if self.reactions is not None else -1

    def to_dict(self) -> Dict[str, Any]:
        payload: dict[str, Any] = {
            "product": self.product,
            "synthesizable": self.synthesizable,
            "reactions": list(self.reactions) if self.reactions is not None else None,
            "length": self.length,
            "source": self.source.to_dict() if self.source is not None else None,
        }
        return {self.route_id: payload}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Route":
        route_id, payload = next(iter(data.items()))
        payload = dict(payload)
        source_data = payload.pop("source", None)
        payload.pop("length", None)
        return cls(
            route_id=route_id,
            product=payload.pop("product"),
            synthesizable=payload.pop("synthesizable"),
            reactions=payload.pop("reactions", None),
            source=RouteSource.from_dict(source_data) if source_data else None,
        )

    @classmethod
    def many_from_dict(cls, data: Dict[str, Any]) -> List["Route"]:
        return [cls.from_dict(d) for d in data]
    
    @classmethod
    def many_to_dict(cls, data: List["Route"]) -> Dict[str, Any]:
        mydict = {}
        for d in data:
            mydict.update(cls.to_dict(d))
        return mydict