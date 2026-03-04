"""Calculator for phonon properties under quasi-harmonic approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from phonopy import PhonopyQHA

from ._base import PropCalc
from ._phonon import PhononCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any, Literal

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

# Default output filenames for each write_* option when set to True.
_WRITE_DEFAULTS: dict[str, str] = {
    "write_helmholtz_volume": "helmholtz_volume.dat",
    "write_volume_temperature": "volume_temperature.dat",
    "write_thermal_expansion": "thermal_expansion.dat",
    "write_gibbs_temperature": "gibbs_temperature.dat",
    "write_bulk_modulus_temperature": "bulk_modulus_temperature.dat",
    "write_heat_capacity_p_numerical": "Cp_temperature.dat",
    "write_heat_capacity_p_polyfit": "Cp_temperature_polyfit.dat",
    "write_gruneisen_temperature": "gruneisen_temperature.dat",
}


class QHACalc(PropCalc):
    """Calculator for thermal properties under the quasi-harmonic approximation.

    Uses phonopy and pymatgen to calculate Gibbs free energy, thermal expansion,
    heat capacity, bulk modulus, and Gruneisen parameters as functions of
    temperature.

    Args:
        calculator: ASE Calculator or string name of a universal calculator.
        t_step: Temperature step in K.
        t_max: Maximum temperature in K.
        t_min: Minimum temperature in K.
        pressure: Pressure in GPa, added as a PV term.
        fmax: Force convergence for relaxation in eV/A.
        optimizer: Optimizer for relaxation.
        eos: Equation of state for E-V fitting.
        relax_structure: Whether to relax before phonon calculations.
        relax_calc_kwargs: Extra keyword arguments passed to RelaxCalc.
        phonon_calc_kwargs: Extra keyword arguments passed to PhononCalc.
        scale_factors: Lattice scale factors for volume sampling.
        write_helmholtz_volume: Output path, True for default filename, or False to skip.
        write_volume_temperature: Same convention as write_helmholtz_volume.
        write_thermal_expansion: Same convention as write_helmholtz_volume.
        write_gibbs_temperature: Same convention as write_helmholtz_volume.
        write_bulk_modulus_temperature: Same convention as write_helmholtz_volume.
        write_heat_capacity_p_numerical: Same convention as write_helmholtz_volume.
        write_heat_capacity_p_polyfit: Same convention as write_helmholtz_volume.
        write_gruneisen_temperature: Same convention as write_helmholtz_volume.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        fmax: float = 0.05,
        pressure: None | float = None,
        optimizer: str = "FIRE",
        eos: Literal["vinet", "birch_murnaghan", "murnaghan"] = "vinet",
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        phonon_calc_kwargs: dict | None = None,
        scale_factors: Sequence[float] = tuple(np.arange(0.95, 1.05, 0.01)),
        write_helmholtz_volume: bool | str | Path = False,
        write_volume_temperature: bool | str | Path = False,
        write_thermal_expansion: bool | str | Path = False,
        write_gibbs_temperature: bool | str | Path = False,
        write_bulk_modulus_temperature: bool | str | Path = False,
        write_heat_capacity_p_numerical: bool | str | Path = False,
        write_heat_capacity_p_polyfit: bool | str | Path = False,
        write_gruneisen_temperature: bool | str | Path = False,
    ) -> None:
        self.calculator = calculator  # type: ignore[assignment]
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.pressure = pressure
        self.fmax = fmax
        self.optimizer = optimizer
        self.eos = eos
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs or {}
        self.phonon_calc_kwargs = phonon_calc_kwargs or {}
        self.scale_factors = scale_factors

        # Resolve write_* params: True uses default filename, False stores "".
        write_options = {
            "write_helmholtz_volume": write_helmholtz_volume,
            "write_volume_temperature": write_volume_temperature,
            "write_thermal_expansion": write_thermal_expansion,
            "write_gibbs_temperature": write_gibbs_temperature,
            "write_bulk_modulus_temperature": write_bulk_modulus_temperature,
            "write_heat_capacity_p_numerical": write_heat_capacity_p_numerical,
            "write_heat_capacity_p_polyfit": write_heat_capacity_p_polyfit,
            "write_gruneisen_temperature": write_gruneisen_temperature,
        }
        for key, val in write_options.items():
            if val is True:
                setattr(self, key, _WRITE_DEFAULTS[key])
            elif val is False:
                setattr(self, key, "")
            else:
                setattr(self, key, str(val))

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """Calculate QHA thermal properties for a given structure.

        Accepts a pymatgen Structure, ASE Atoms, or a dict from a previous
        calc() call. The structure is always converted to a pymatgen Structure
        before volume scaling, so ASE Atoms inputs work regardless of the
        relax_structure setting.

        Args:
            structure: Input structure in any supported format.

        Returns:
            Dict with keys: qha, scale_factors, volumes, electronic_energies,
            temperatures, thermal_expansion_coefficients, gibbs_free_energies,
            bulk_modulus_P, heat_capacity_P, gruneisen_parameters.
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator,
                fmax=self.fmax,
                optimizer=self.optimizer,
                **self.relax_calc_kwargs,
            )
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]

        # Convert to pymatgen Structure for apply_strain; needed when
        # relax_structure is False and input is ASE Atoms.
        structure_in = to_pmg_structure(structure_in)

        # Create PhononCalc once and reuse across all scale factors.
        phonon_calc = PhononCalc(
            self.calculator,
            t_step=self.t_step,
            t_max=self.t_max,
            t_min=self.t_min,
            relax_structure=False,
            write_phonon=False,
            **self.phonon_calc_kwargs,
        )

        volumes = []
        electronic_energies = []
        free_energies = []
        entropies = []
        heat_capacities = []

        for factor in self.scale_factors:
            scaled = structure_in.copy()
            scaled.apply_strain(factor - 1)

            volumes.append(scaled.volume)
            electronic_energies.append(run_pes_calc(scaled, self.calculator).energy)

            thermal = phonon_calc.calc(scaled)["thermal_properties"]
            free_energies.append(thermal["free_energy"])
            entropies.append(thermal["entropy"])
            heat_capacities.append(thermal["heat_capacity"])

        temperatures = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)

        qha = PhonopyQHA(
            volumes=volumes,
            electronic_energies=electronic_energies,
            temperatures=temperatures,
            free_energy=np.transpose(free_energies),
            cv=np.transpose(heat_capacities),
            entropy=np.transpose(entropies),
            pressure=self.pressure,
            eos=self.eos,
            t_max=self.t_max,
        )

        self._write_output_files(qha)

        # PhonopyQHA clips output arrays by one element (the last temperature
        # point is consumed by finite-difference derivatives). Extract the
        # matching temperature array so all outputs have equal length.
        qha_core = qha._qha  # noqa: SLF001
        qha_temperatures = qha_core._temperatures[: qha_core._len]  # noqa: SLF001

        return result | {
            "qha": qha,
            "scale_factors": self.scale_factors,
            "volumes": volumes,
            "electronic_energies": electronic_energies,
            "temperatures": qha_temperatures,
            "thermal_expansion_coefficients": qha.thermal_expansion,
            "gibbs_free_energies": qha.gibbs_temperature,
            "bulk_modulus_P": qha.bulk_modulus_temperature,
            "heat_capacity_P": qha.heat_capacity_P_polyfit,
            "gruneisen_parameters": qha.gruneisen_temperature,
        }

    def _write_output_files(self, qha: PhonopyQHA) -> None:
        """Write QHA results to files for any non-empty write_* attributes.

        Maps each matcalc attribute name to the corresponding PhonopyQHA
        writer method. The naming differs for heat capacity methods
        (lowercase "p" in matcalc vs uppercase "P" in phonopy).
        """
        writers: dict[str, str] = {
            "write_helmholtz_volume": "write_helmholtz_volume",
            "write_volume_temperature": "write_volume_temperature",
            "write_thermal_expansion": "write_thermal_expansion",
            "write_gibbs_temperature": "write_gibbs_temperature",
            "write_bulk_modulus_temperature": "write_bulk_modulus_temperature",
            "write_heat_capacity_p_numerical": "write_heat_capacity_P_numerical",
            "write_heat_capacity_p_polyfit": "write_heat_capacity_P_polyfit",
            "write_gruneisen_temperature": "write_gruneisen_temperature",
        }
        for attr, method_name in writers.items():
            filename = getattr(self, attr)
            if filename:
                getattr(qha, method_name)(filename=filename)
