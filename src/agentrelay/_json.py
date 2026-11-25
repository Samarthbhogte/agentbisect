"""Convert arbitrary objects (SDK models, tuples, datetimes) to plain JSON-ready values."""

from __future__ import annotations

import base64
from datetime import date, datetime
from typing import Any


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return {"$bytes": base64.b64encode(obj).decode("ascii")}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return to_jsonable(dump(mode="json"))
        except TypeError:
            return to_jsonable(dump())
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)
