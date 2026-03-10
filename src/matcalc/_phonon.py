"""Calculator for phonon properties."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS as write_force_constants
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Literal

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class PhononCalc(PropCalc):
    """
    A class for phonon and thermal property calculations using phonopy.

    The `PhononCalc` class extends the `PropCalc` class to provide
    functionalities for calculating phonon properties and thermal properties
    of a given structure using the phonopy library. It includes options for
    structural relaxation before phonon property determination, as well as
    methods to export calculated properties to various output files for
    further analysis or visualization.

    :ivar calculator: A calculator object or a string specifying the
        computational backend to be used.
    :type calculator: Calculator | str
    :ivar atom_disp: Magnitude of atomic displacements for phonon
        calculations.
    :type atom_disp: float
    :ivar min_length: Minimum length of lattice dimensions, used to
        automatically set supercell_matrix.
    :type min_length: float | None
    :ivar supercell_matrix: Array defining the transformation matrix to
        construct supercells for phonon calculations. Takes precedence
        over min_length.
    :type supercell_matrix: ArrayLike | None
    :ivar t_step: Temperature step for thermal property calculations in
        Kelvin.
    :type t_step: float
    :ivar t_max: Maximum temperature for thermal property calculations in
        Kelvin.
    :type t_max: float
    :ivar t_min: Minimum temperature for thermal property calculations in
        Kelvin.
    :type t_min: float
    :ivar fmax: Maximum force convergence criterion for structural relaxation.
    :type fmax: float
    :ivar optimizer: String specifying the optimizer type to be used for
        structural relaxation.
    :type optimizer: str
    :ivar relax_structure: Boolean flag to determine whether to relax the
        structure before phonon calculation.
    :type relax_structure: bool
    :ivar relax_calc_kwargs: Optional dictionary containing additional
        arguments for the structural relaxation calculation.
    :type relax_calc_kwargs: dict | None
    :ivar imaginary_freq_tol: Tolerance for imaginary frequency detection in
        THz. If a frequency is found with a value below imaginary_freq_tol,
        it is considered imaginary.
    :type imaginary_freq_tol: float
    :ivar on_imaginary_modes: If there is a frequency with a value below
        imaginary_freq_tol, then either raise a ValueError ("error") or log a
        warning ("warn").
    :type on_imaginary_modes: Literal["error", "warn"]
    :ivar fix_imaginary_attempts: Number of attempts to resolve imaginary modes by rattling
        the involved atoms (ASE rattle, stdev=0.01 Å) and re-optimizing at fixed cell volume.
        0 disables correction.
    :type fix_imaginary_attempts: int
    :ivar write_force_constants: Path, boolean, or string specifying whether
        to write the calculated force constants to an output file, and the
        path or name of the file if applicable.
    :type write_force_constants: bool | str | Path
    :ivar write_band_structure: Path, boolean, or string specifying whether
        to write the calculated phonon band structure to an output file,
        and the path or name of the file if applicable.
    :type write_band_structure: bool | str | Path
    :ivar write_total_dos: Path, boolean, or string specifying whether to
        write the calculated total density of states (DOS) to an output
        file, and the path or name of the file if applicable.
    :type write_total_dos: bool | str | Path
    :ivar write_phonon: Path, boolean, or string specifying whether to write
        the calculated phonon properties (e.g., phonon.yaml) to an output
        file, and the path or name of the file if applicable.
    :type write_phonon: bool | str | Path
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        atom_disp: float = 0.015,
        min_length: float | None = 20.0,
        supercell_matrix: ArrayLike | None = None,
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        fmax: float = 1e-5,
        max_steps: int = 5000,
        optimizer: str = "FIRE",
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        imaginary_freq_tol: float = -0.01,
        on_imaginary_modes: Literal["error", "warn"] = "warn",
        fix_imaginary_attempts: int = 0,
        write_force_constants: bool | str | Path = False,
        write_band_structure: bool | str | Path = False,
        write_total_dos: bool | str | Path = False,
        write_phonon: bool | str | Path = True,
    ) -> None:
        """
        Initializes the class with configuration for the phonon calculations. The initialization parameters control
        the behavior of structural relaxation, thermal properties, force calculations, and output file generation.
        The class allows for customization of key parameters to facilitate the study of material behaviors.

        :param calculator: The calculator object or string name specifying the calculation backend to use.
        :param atom_disp: Atom displacement to be used for finite difference calculation of force constants.
        :param min_length: Minimum length of lattice dimensions, used to automatically set supercell_matrix.
        :param supercell_matrix: Transformation matrix to define the supercell for the calculation. Takes
            precedence over min_length
        :param t_step: Temperature step for thermal property calculations.
        :param t_max: Maximum temperature for thermal property calculations.
        :param t_min: Minimum temperature for thermal property calculations.
        :param fmax: Maximum force during structure relaxation, used as a convergence criterion.
        :param max_steps: The maximum number of optimization steps to perform during the relaxation process.
        :param optimizer: Name of the optimization algorithm for structural relaxation.
        :param relax_structure: Flag to indicate whether structure relaxation should be performed before calculations.
        :param relax_calc_kwargs: Additional keyword arguments for relaxation phase calculations.
        :param imaginary_freq_tol: Tolerance for imaginary frequency detection in THz. If a frequency is found with
            a value below imaginary_freq_tol, it is considered imaginary.
        :param on_imaginary_modes: If there is a frequency with a value below imaginary_freq_tol, then
            raise a ValueError ("error") or log a warning ("warn"). Defaults to "warn".
        :param fix_imaginary_attempts: Number of attempts to resolve imaginary modes. For each attempt, atoms
            involved in imaginary modes are rattled (ASE rattle, stdev=0.01 Å), the structure is re-optimized
            at fixed cell volume, and phonons are recalculated. 0 disables correction.
        :param write_force_constants: File path or boolean flag to write force constants.
            Defaults to "force_constants".
        :param write_band_structure: File path or boolean flag to write band structure data.
            Defaults to "band_structure.yaml".
        :param write_total_dos: File path or boolean flag to write total density of states (DOS) data.
            Defaults to "total_dos.dat".
        :param write_phonon: File path or boolean flag to write phonon data. Defaults to "phonon.yaml".
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.atom_disp = atom_disp
        self.min_length = min_length
        self.supercell_matrix = supercell_matrix
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        self.imaginary_freq_tol = imaginary_freq_tol
        self.on_imaginary_modes = on_imaginary_modes
        self.fix_imaginary_attempts = fix_imaginary_attempts
        self.write_force_constants = write_force_constants
        self.write_band_structure = write_band_structure
        self.write_total_dos = write_total_dos
        self.write_phonon = write_phonon

        if supercell_matrix is None and min_length is None:
            raise ValueError("min_length must be set when supercell_matrix is None.")

        # Set default paths for output files.
        for key, val, default_path in (
            ("write_force_constants", self.write_force_constants, "force_constants"),
            ("write_band_structure", self.write_band_structure, "band_structure.yaml"),
            ("write_total_dos", self.write_total_dos, "total_dos.dat"),
            ("write_phonon", self.write_phonon, "phonon.yaml"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """Calculates thermal properties of Pymatgen structure with phonopy.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            phonon: Phonopy object with force constants produced
            thermal_properties:
                {
                    temperatures: list of temperatures in Kelvin,
                    free_energy: list of Helmholtz free energies at corresponding temperatures in kJ/mol,
                    entropy: list of entropies at corresponding temperatures in J/K/mol,
                    heat_capacity: list of heat capacities at constant volume at corresponding temperatures in J/K/mol,
                    The units are originally documented in phonopy.
                    See phonopy.Phonopy.run_thermal_properties()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/api_phonopy.py#L2591)
                    -> phonopy.phonon.thermal_properties.ThermalProperties.run()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L498)
                    -> phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_free_energy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L217)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_entropy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L233)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_heat_capacity()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L225)
                }
        }
        """
        result = super().calc(structure)
        structure_in: Structure = to_pmg_structure(result["final_structure"])

        if self.relax_structure:
            logger.info("Relaxing input structure before phonon calculation")
            relax_calc_kwargs = {"fmax": self.fmax, "optimizer": self.optimizer, "max_steps": self.max_steps} | (
                self.relax_calc_kwargs or {}
            )
            relaxer = RelaxCalc(self.calculator, **relax_calc_kwargs)
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]

        phonon, frequencies, disp_supercells = self._run_phonopy(structure_in)

        if self.fix_imaginary_attempts > 0 and np.any(frequencies < self.imaginary_freq_tol):
            phonon, frequencies, disp_supercells, relax_result = self._resolve_imaginary_modes(
                structure_in, phonon, frequencies, disp_supercells
            )
            result |= relax_result
            structure_in = result["final_structure"]

        self._check_imaginary_modes(frequencies)

        phonon.run_thermal_properties(t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)
        if self.write_force_constants:
            write_force_constants(phonon.force_constants, filename=self.write_force_constants)  # type: ignore[arg-type]
        if self.write_band_structure:
            phonon.auto_band_structure(write_yaml=True, filename=self.write_band_structure)
        if self.write_total_dos:
            phonon.auto_total_dos(write_dat=True, filename=self.write_total_dos)
        if self.write_phonon:
            phonon.save(filename=self.write_phonon)  # type: ignore[arg-type]
        return result | {
            "phonon": phonon,
            "thermal_properties": phonon.get_thermal_properties_dict(),
            "frequencies": frequencies,
            "disp_supercells": disp_supercells,
        }

    def _run_phonopy(self, structure: Structure) -> tuple[phonopy.Phonopy, np.ndarray, list[Structure]]:
        """Build a Phonopy object, run force calculations, and return (phonon, frequencies, disp_supercells).

        Args:
            structure: Pymatgen structure.

        Returns:
            Tuple of (phonon, frequencies, disp_supercells).
        """
        cell = get_phonopy_structure(structure)
        if self.supercell_matrix:
            supercell_matrix = np.array(self.supercell_matrix, dtype=int)
        else:
            supercell_matrix = np.diag(np.ceil(self.min_length / np.array(structure.lattice.abc)).astype(int))

        phonon = phonopy.Phonopy(cell, supercell_matrix=supercell_matrix)
        phonon.generate_displacements(distance=self.atom_disp)
        disp_supercells = [
            get_pmg_structure(supercell)
            for supercell in phonon.supercells_with_displacements  # type:ignore[union-attr]
            if supercell is not None
        ]
        logger.info("Computing forces on %d displaced supercells", len(disp_supercells))
        phonon.forces = [run_pes_calc(supercell, self.calculator).forces for supercell in disp_supercells]
        phonon.produce_force_constants()
        phonon.run_mesh()
        frequencies = phonon.get_mesh_dict()["frequencies"]
        return phonon, frequencies, disp_supercells

    def _check_imaginary_modes(self, frequencies: np.ndarray) -> None:
        """Warn or raise if imaginary modes are present, based on on_imaginary_modes setting.

        Args:
            frequencies: Frequencies array from the mesh run.
        """
        # In phonopy, imaginary frequencies are represented as negative values.
        imag_freq_mask = frequencies < self.imaginary_freq_tol
        if np.any(imag_freq_mask):
            n_imag = np.sum(imag_freq_mask)
            n_freqs = frequencies.size
            min_mode = np.min(frequencies)
            pct = 100 * n_imag / n_freqs
            msg = (
                f"{n_imag}/{n_freqs} ({pct:.12}%) modes are imaginary (below {self.imaginary_freq_tol:.4f} THz). "
                f"Most negative: {min_mode:.2f} THz. This indicates a dynamically unstable structure. "
                f"Thermal properties may not be reliable."
            )
            if self.on_imaginary_modes.lower() == "warn":
                logger.warning(msg)
            else:
                raise ValueError(msg)

    def _resolve_imaginary_modes(
        self,
        structure_in: Structure,
        phonon: phonopy.Phonopy,
        frequencies: np.ndarray,
        disp_supercells: list,
    ) -> tuple[phonopy.Phonopy, np.ndarray, list[Structure], dict]:
        """Attempt to resolve imaginary modes by rattling and re-relaxing.

        Args:
            structure_in: Current primitive-cell structure.
            phonon: Phonopy object from the most recent build.
            frequencies: Frequencies from the most recent mesh run.
            disp_supercells: Displaced supercells from the most recent build.

        Returns:
            Updated (phonon, frequencies, disp_supercells, relax_result).
        """
        logger.warning(
            "Imaginary modes detected (min freq: %.1f THz). Attempting to resolve over %d attempt(s).",
            np.min(frequencies),
            self.fix_imaginary_attempts,
        )
        relax_result: dict = {}
        for attempt in range(self.fix_imaginary_attempts):
            logger.info("Imaginary mode correction attempt %d/%d", attempt + 1, self.fix_imaginary_attempts)
            involved = self._get_imaginary_mode_atom_indices(phonon, structure_in)

            logger.info("Rattling %d/%d atoms involved in imaginary modes", len(involved), len(structure_in))
            structure_in = self._rattle_structure(structure_in, involved)

            logger.info("Re-relaxing structure at fixed cell volume")
            relax_result = self._relax_fixed_cell(structure_in)
            structure_in = relax_result["final_structure"]

            logger.info("Recomputing phonons after correction attempt %d", attempt + 1)
            phonon, frequencies, disp_supercells = self._run_phonopy(structure_in)

            if not np.any(frequencies < self.imaginary_freq_tol):
                logger.info("Imaginary modes resolved after %d attempt(s)", attempt + 1)
                break
            logger.info(
                "Imaginary modes persist after attempt %d (min freq: %.1f THz).",
                attempt + 1,
                np.min(frequencies),
            )
        else:
            logger.warning(
                "Imaginary modes not resolved after %d attempt(s) (min freq: %.1f THz).",
                self.fix_imaginary_attempts,
                np.min(frequencies),
            )
        return phonon, frequencies, disp_supercells, relax_result

    def _get_imaginary_mode_atom_indices(self, phonon: phonopy.Phonopy, structure_in: Structure) -> np.ndarray:
        """Return indices of primitive-cell atoms with significant amplitude in imaginary modes.

        Args:
            phonon: Phonopy object with force constants already computed.
            structure_in: Pymatgen structure (primitive cell).

        Returns:
            Array of atom indices involved in imaginary modes.
        """
        phonon.run_mesh(with_eigenvectors=True)
        mesh_dict = phonon.get_mesh_dict()
        eigenvectors = mesh_dict["eigenvectors"]  # shape (nqpoints, nbands, nbands) complex
        freqs = mesh_dict["frequencies"]  # shape (nqpoints, nbands)

        # Select imaginary-mode eigenvectors and reshape to (n_imag, natoms, 3),
        # then sum squared amplitudes over all imaginary modes and Cartesian directions
        natoms = len(structure_in)
        imag_evecs = eigenvectors[freqs < self.imaginary_freq_tol]  # (n_imag, 3*natoms)
        atom_amplitudes = np.sum(np.abs(imag_evecs.reshape(-1, natoms, 3)) ** 2, axis=(0, 2))
        return np.where(atom_amplitudes >= 0.1 * atom_amplitudes.max())[0]

    def _rattle_structure(self, structure_in: Structure, involved_atoms: np.ndarray) -> Structure:
        """Rattle only the involved atoms using ASE rattle (stdev=0.01 Å).

        Args:
            structure_in: Pymatgen structure.
            involved_atoms: Indices of atoms to rattle.

        Returns:
            Pymatgen structure with rattled atomic positions.
        """
        atoms = to_ase_atoms(structure_in)
        original_positions = atoms.positions.copy()
        atoms.rattle(stdev=0.01)
        uninvolved = np.setdiff1d(np.arange(len(atoms)), involved_atoms)
        atoms.positions[uninvolved] = original_positions[uninvolved]
        return to_pmg_structure(atoms)

    def _relax_fixed_cell(self, structure_in: Structure) -> dict:
        """Relax atomic positions at fixed cell.

        Args:
            structure_in: Pymatgen structure.

        Returns:
            Full result dict from RelaxCalc.
        """
        relaxer = RelaxCalc(
            self.calculator,
            fmax=self.fmax,
            optimizer=self.optimizer,
            max_steps=self.max_steps,
            relax_cell=False,  # changing this to True will break QHACalc
        )
        return relaxer.calc(structure_in)
