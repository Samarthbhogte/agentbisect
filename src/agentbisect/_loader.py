"""Load user Python files (oracles, FakeLLM rules) given as paths or module names."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_module(ref: str) -> ModuleType:
    """`path/to/file.py` or `package.module`.

    A file inside a package (every parent up to the working directory has `__init__.py`) is
    imported under its dotted name so its relative imports work.
    """
    if not ref.endswith(".py"):
        return importlib.import_module(ref)
    path = Path(ref).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {ref}")
    dotted = _dotted_name(path)
    if dotted is not None:
        root = str(Path.cwd())
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module(dotted)
    name = f"agentbisect_user_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {ref}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dotted_name(path: Path) -> str | None:
    cwd = Path.cwd().resolve()
    try:
        rel = path.relative_to(cwd)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if len(parts) < 2:
        return None
    for i in range(1, len(parts)):
        if not (cwd.joinpath(*parts[:i]) / "__init__.py").is_file():
            return None
    return ".".join(parts)


def load_attr(ref: str) -> Any:
    """`file.py:name` or `package.module:name` -> the object."""
    module_ref, sep, attr = ref.rpartition(":")
    if not sep or not module_ref or not attr:
        raise ValueError(f"expected FILE:NAME or MODULE:NAME, got {ref!r}")
    module = load_module(module_ref)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"{module_ref} has no attribute {attr!r}") from None
