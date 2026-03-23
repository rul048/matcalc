"""
This module implements convenience functions to perform various atomic simulation types, specifically relaxation and
static, using either ASE or LAMMPS. This enables the code to use either for the computation of various properties.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from inspect import isclass
from typing import TYPE_CHECKING

import ase
import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import Trajectory
from ase.optimize.optimize import Optimizer

from matcalc.utils import to_ase_atoms, to_pmg_structure

from ._base import SimulationResult

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


def is_ase_optimizer(key: str | Optimizer) -> bool:
    """
    Determines whether the given key is an ASE optimizer. A key can
    either be a string representing the name of an optimizer class
    within `ase.optimize` or directly be an optimizer class that
    subclasses `Optimizer`.

    If the key is a string, the function checks whether it corresponds
    to a class in `ase.optimize` that is a subclass of `Optimizer`.

    :param key: The key to check, either a string name of an ASE
        optimizer class or a class object that potentially subclasses
        `Optimizer`.
    :return: True if the key is either a string corresponding to an
        ASE optimizer subclass name in `ase.optimize` or a class that
        is a subclass of `Optimizer`. Otherwise, returns False.
    """
    if isclass(key) and issubclass(key, Optimizer):
        return True
    if isinstance(key, str):
        return isclass(obj := getattr(ase.optimize, key, None)) and issubclass(obj, Optimizer)
    return False


VALID_OPTIMIZERS = [key for key in dir(ase.optimize) if is_ase_optimizer(key)]


def get_ase_optimizer(optimizer: str | Optimizer) -> Optimizer:
    """
    Retrieve an ASE optimizer instance based on the provided input. This function accepts either a
    string representing the name of a valid ASE optimizer or an instance/subclass of the Optimizer
    class. If a string is provided, it checks the validity of the optimizer name, and if valid, retrieves
    the corresponding optimizer from ASE. An error is raised if the optimizer name is invalid.

    If an Optimizer subclass or instance is provided as input, it is returned directly.

    :param optimizer: The optimizer to be retrieved. Can be a string representing a valid ASE
        optimizer name or an instance/subclass of the Optimizer class.
    :type optimizer: str | Optimizer

    :return: The corresponding ASE optimizer instance or the input Optimizer instance/subclass.
    :rtype: Optimizer

    :raises ValueError: If the optimizer name provided as a string is not among the valid ASE
        optimizer names defined by `VALID_OPTIMIZERS`.
    """
    if isclass(optimizer) and issubclass(optimizer, Optimizer):
        return optimizer

    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(f"Unknown {optimizer=}, must be one of {VALID_OPTIMIZERS}")

    return getattr(ase.optimize, optimizer) if isinstance(optimizer, str) else optimizer


def run_ase(
    structure: Structure | Atoms,
    calculator: Calculator,
    *,
    relax_atoms: bool = False,
    relax_cell: bool = False,
    optimizer: Optimizer | str = "FIRE",
    max_steps: int = 500,
    traj_file: str | None = None,
    interval: int = 1,
    fmax: float = 0.1,
    cell_filter: Filter = FrechetCellFilter,  # type:ignore[assignment]
    cell_filter_kwargs: dict | None = None,
) -> SimulationResult:
    """
    Run ASE static calculation using the given structure and calculator.

    Parameters:
    structure (Structure|Atoms): The input structure to calculate potential energy, forces, and stress.
    calculator (Calculator): The calculator object to use for the calculation.

    Returns:
    PESResult: Object containing potential energy, forces, and stress of the input structure.
    """
    atoms = to_ase_atoms(structure)
    atoms.calc = calculator
    cell_filter_kwargs = cell_filter_kwargs or {}
    if relax_atoms:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            original_atoms = atoms
            if relax_cell:
                atoms = cell_filter(atoms, **cell_filter_kwargs)  # type:ignore[operator]
            opt = get_ase_optimizer(optimizer)(atoms)  # type:ignore[operator]
            traj = Trajectory(traj_file, "w", original_atoms) if traj_file is not None else None
            if traj is not None:
                opt.attach(traj, interval=interval)
            opt.run(fmax=fmax, steps=max_steps)
            if traj is not None:
                traj.close()
            if opt.nsteps >= max_steps:
                warnings.warn("Maximum steps reached in structure relaxation.", UserWarning, stacklevel=2)

        if relax_cell:
            atoms = atoms.atoms  # type:ignore[attr-defined]
        if opt.nsteps >= max_steps:
            max_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
            logger.warning(
                "Maximum steps reached in structure relaxation. Max|F| = %s eV/Å",
                max_force,
            )

        return SimulationResult(
            to_pmg_structure(atoms),
            atoms.get_potential_energy(),
            atoms.get_kinetic_energy(),
            atoms.get_total_energy(),
            atoms.get_forces(),
            atoms.get_stress(voigt=False),
        )

    return SimulationResult(
        to_pmg_structure(structure),
        atoms.get_potential_energy(),
        atoms.get_kinetic_energy(),
        atoms.get_total_energy(),
        atoms.get_forces(),
        atoms.get_stress(voigt=False),
    )
