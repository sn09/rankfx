"""Module with utils for importing packages."""

import importlib
from typing import Any


def import_module_by_path(import_path: str) -> Any:
    """Import any object by its path.

    Args:
        import_path: path to object

    Returns:
        Imported object
    """
    module_name, object_name = import_path.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module_name), object_name)
