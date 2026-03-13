"""Relaxation properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry, FixAtoms

from ._base import PropCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure, to_ase_atoms

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class RelaxCalc(PropCalc):
    """
    A class to perform structural relaxation calculations using ASE (Atomic Simulation Environment).

    This class facilitates structural relaxation by integrating with ASE tools for
    optimized geometry and/or cell parameters. It enables convergence on forces, stress,
    and total energy, offering customization for relaxation parameters and further
    enabling properties to be extracted from the relaxed structure.

    :ivar calculator: ASE Calculator used for force and energy evaluations.
    :type calculator: Calculator

    :ivar optimizer: Algorithm for performing the optimization.
    :type optimizer: Optimizer | str

    :ivar max_steps: Maximum number of optimization steps allowed.
    :type max_steps: int

    :ivar traj_file: Path to save relaxation trajectory (optional).
    :type traj_file: str | None

    :ivar interval: Interval for saving trajectory frames during relaxation.
    :type interval: int

    :ivar fmax: Force tolerance for convergence (eV/Å).
    :type fmax: float

    :ivar relax_atoms: Whether atomic positions are relaxed.
    :type relax_atoms: bool

    :ivar relax_cell: Whether cell parameters are relaxed.
    :type relax_cell: bool

    :ivar cell_filter: ASE filter used for modifying the cell during relaxation.
    :type cell_filter: Filter

    :ivar cell_filter_kwargs: The keyword arguments to pass to the cell_filter.
    :type cell_filter_kwargs: dict | None

    :ivar perturb_distance: Distance (Å) for random perturbation to break symmetry.
    :type perturb_distance: float | None

    :ivar fix_symmetry: Whether symmetry is preserved.
    :type fix_symmetry: bool

    :ivar fix_atoms: Whether all atoms are fixed; if list/tuple, fixes the given indices.
    :type fix_atoms: bool | list[int]

    :ivar symprec: Tolerance for symmetry constraints.
    :type symprec: float
    """

    def __init__(
        self,
        calculator: "Calculator | str",
        *,
        optimizer: "Optimizer | str" = "FIRE",
        max_steps: int = 500,
        traj_file: "str | None" = None,
        interval: int = 1,
        fmax: float = 0.1,
        relax_atoms: bool = True,
        relax_cell: bool = True,
        cell_filter: "Filter" = FrechetCellFilter,  # type: ignore[assignment]
        cell_filter_kwargs: "dict | None" = None,
        perturb_distance: "float | None" = None,
        fix_symmetry: bool = False,
        fix_atoms: "bool | list[int]" = False,
        symprec: float = 1e-3,
    ) -> None:
        """
        Initializes the relaxation procedure for an atomic configuration system.

        :param calculator: ASE Calculator used for force and energy evaluations.
        :param optimizer: ASE optimizer name or class (e.g., "FIRE", BFGS).
        :param max_steps: Maximum optimization steps.
        :param traj_file: Trajectory file (optional).
        :param interval: Interval for trajectory saving.
        :param fmax: Force convergence threshold (eV/Å).
        :param relax_atoms: Whether atomic positions are relaxed.
        :param relax_cell: Whether cell parameters are relaxed.
        :param cell_filter: ASE cell filter used when relaxing the cell.
        :param cell_filter_kwargs: kwargs for the cell_filter.
        :param perturb_distance: Optional random perturbation distance (Å).
        :param fix_symmetry: Whether symmetry is preserved.
        :param fix_atoms: Whether all atoms are fixed; if list/tuple, fixes the given indices.
        :param symprec: Tolerance for symmetry constraints.
        """
        self.calculator = calculator  # type: ignore[assignment]

        self.optimizer = optimizer
        self.fmax = fmax
        self.interval = interval
        self.max_steps = max_steps
        self.traj_file = traj_file
        self.relax_cell = relax_cell
        self.relax_atoms = relax_atoms
        self.cell_filter = cell_filter
        self.cell_filter_kwargs = cell_filter_kwargs
        self.perturb_distance = perturb_distance
        self.fix_symmetry = fix_symmetry
        self.fix_atoms = fix_atoms
        self.symprec = symprec

    def calc(self, structure: "Structure | Atoms | dict[str, Any]") -> dict:
        """
        Calculate the final relaxed structure, energy, forces, and stress for a given
        structure and update the result dictionary with additional geometric properties.

        This method takes an input structure and performs a relaxation process
        depending on the specified parameters. If the `perturb_distance` attribute
        is provided, the structure is perturbed before relaxation. The relaxation
        process can involve optimization of both atomic positions and the unit cell
        if specified. Results of the calculation including final structure geometry,
        energy, forces, stresses, and lattice parameters are returned in a dictionary.

        :param structure: Input structure for calculation. Can be provided as a
                          `Structure` object or a dictionary convertible to `Structure`.
        :return: Dictionary containing the final relaxed structure, calculated
                 energy, forces, stress, and lattice parameters.
        :rtype: dict
        """
        result = super().calc(structure)

        structure_in: "Structure | Atoms" = result["final_structure"]

        if self.perturb_distance is not None:
            structure_in = to_pmg_structure(structure_in).perturb(
                distance=self.perturb_distance, seed=None
            )

        structure_in = to_ase_atoms(structure_in)
        constraints = []
        if self.fix_atoms:
            if isinstance(self.fix_atoms, bool):
                indices = list(range(len(structure_in)))
            elif isinstance(self.fix_atoms, (list, tuple)):
                indices = [int(i) for i in self.fix_atoms]
            constraints.append(FixAtoms(indices=indices))

        if self.fix_symmetry:
            constraints.append(FixSymmetry(structure_in, symprec=self.symprec))

        if constraints:
            structure_in.set_constraint(constraints)

        r = run_pes_calc(
            structure_in,
            self.calculator,
            relax_atoms=self.relax_atoms,
            relax_cell=self.relax_cell,
            optimizer=self.optimizer,
            max_steps=self.max_steps,
            traj_file=self.traj_file,
            interval=self.interval,
            fmax=self.fmax,
            cell_filter=self.cell_filter,
            cell_filter_kwargs=self.cell_filter_kwargs,
        )

        lattice = r.structure.lattice
        result.update(
            {
                "final_structure": r.structure,
                "energy": r.potential_energy,
                "forces": r.forces,
                "stress": r.stress,
                "a": lattice.a,
                "b": lattice.b,
                "c": lattice.c,
                "alpha": lattice.alpha,
                "beta": lattice.beta,
                "gamma": lattice.gamma,
                "volume": lattice.volume,
            }
        )

        return result
