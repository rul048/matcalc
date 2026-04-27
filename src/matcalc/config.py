"""Sets some configuration global variables and locations for matcalc."""

from __future__ import annotations

import os
import pathlib
import shutil

BENCHMARK_HF_REPO_ID: str = "materialyze/matcalc-bench"
BENCHMARK_DATA_DIR: pathlib.Path = pathlib.Path.home() / ".cache" / "matcalc"
SIMULATION_BACKEND: str = os.environ.get("MATCALC_BACKEND", "ASE").upper()


def clear_cache(*, confirm: bool = True) -> None:
    """
    Deletes all files and subdirectories within the benchmark data directory,
    effectively clearing the cache. The user is prompted for confirmation
    before proceeding with the deletion to prevent accidental data loss.

    Args:
        confirm: If True (default), prompt before deleting. If False, delete without prompting.
    """
    answer = "" if confirm else "y"
    while answer not in ("y", "n"):
        answer = input(f"Do you really want to delete everything in {BENCHMARK_DATA_DIR} (y|n)? ").lower().strip()
    if answer == "y":
        try:
            shutil.rmtree(BENCHMARK_DATA_DIR)
        except FileNotFoundError:
            print(f"matcalc cache dir {BENCHMARK_DATA_DIR} not found")  # noqa: T201
