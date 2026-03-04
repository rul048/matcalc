"""Tests for QHACalc class."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from matcalc import QHACalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure

# All keys that calc() should return
EXPECTED_RESULT_KEYS = {
    "final_structure",
    "qha",
    "scale_factors",
    "volumes",
    "electronic_energies",
    "temperatures",
    "thermal_expansion_coefficients",
    "gibbs_free_energies",
    "bulk_modulus_P",
    "heat_capacity_P",
    "gruneisen_parameters",
}

# Shared test parameters
SCALE_FACTORS = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]
T_STEP = 50
T_MAX = 1000

EXPECTED_VOLUMES = [23.07207, 23.79302, 24.52884, 25.27967, 26.04567, 26.82699, 27.62378]
EXPECTED_ENERGIES = [
    -14.043658256530762,
    -14.065637588500977,
    -14.07603645324707,
    -14.07599925994873,
    -14.066689491271973,
    -14.048959732055664,
    -14.023341178894043,
]


def _assert_output_consistency(result: dict) -> None:
    """Common assertions that every QHA result must satisfy."""
    # 1. All expected keys present
    assert set(result) >= EXPECTED_RESULT_KEYS, f"Missing keys: {EXPECTED_RESULT_KEYS - set(result)}"

    # 2. temperatures, gibbs, thermal_expansion, bulk_modulus, Cp, gruneisen
    #    must all have the same length (Bug #2 regression guard)
    n = len(result["temperatures"])
    assert n > 0, "temperatures should not be empty"
    for key in (
        "thermal_expansion_coefficients",
        "gibbs_free_energies",
        "bulk_modulus_P",
        "heat_capacity_P",
        "gruneisen_parameters",
    ):
        assert len(result[key]) == n, f"Length mismatch: temperatures has {n} but {key} has {len(result[key])}"

    # 3. volumes and electronic_energies must match scale_factors length
    n_scales = len(result["scale_factors"])
    assert len(result["volumes"]) == n_scales
    assert len(result["electronic_energies"]) == n_scales

    # 4. qha object should be a PhonopyQHA
    from phonopy import PhonopyQHA

    assert isinstance(result["qha"], PhonopyQHA)


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


class TestQHACalcBasic:
    """Basic QHA calculation with Structure input and relaxation."""

    def test_volumes_and_energies(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Test that volumes and energies match expected values."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        result = qha_calc.calc(Li2O)

        _assert_output_consistency(result)

        assert result["volumes"] == pytest.approx(EXPECTED_VOLUMES, rel=1e-3)
        assert result["electronic_energies"] == pytest.approx(EXPECTED_ENERGIES, abs=1e-2)

    def test_thermal_properties_at_300K(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Test thermal properties at 300 K."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        result = qha_calc.calc(Li2O)

        ind = result["temperatures"].tolist().index(300)
        assert result["thermal_expansion_coefficients"][ind] == pytest.approx(1.03973e-04, rel=1e-1)
        assert result["gibbs_free_energies"][ind] == pytest.approx(-14.04472, rel=1e-1)
        assert result["bulk_modulus_P"][ind] == pytest.approx(54.25954, rel=1e-1)
        assert result["heat_capacity_P"][ind] == pytest.approx(62.27455, rel=1e-1)
        assert result["gruneisen_parameters"][ind] == pytest.approx(1.688877575687573, rel=1e-1)

    def test_scale_factors_passthrough(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Scale factors in output should match those passed in."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        result = qha_calc.calc(Li2O)
        assert list(result["scale_factors"]) == SCALE_FACTORS


# ---------------------------------------------------------------------------
# Pressure
# ---------------------------------------------------------------------------


class TestQHACalcPressure:
    """QHA with non-zero pressure."""

    def test_pressure_shifts_gibbs(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Gibbs free energy should differ from zero-pressure case."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            pressure=10.0,
        )
        result = qha_calc.calc(Li2O)

        _assert_output_consistency(result)

        # Volumes/energies are from the same structure (before QHA applies PV)
        assert result["volumes"] == pytest.approx(EXPECTED_VOLUMES, abs=1e-1)
        assert result["electronic_energies"] == pytest.approx(EXPECTED_ENERGIES, abs=1e-2)

        # Gibbs at 300 K should reflect the PV term
        ind = result["temperatures"].tolist().index(300)
        assert result["gibbs_free_energies"][ind] == pytest.approx(-12.43074657443325, rel=1e-1)


# ---------------------------------------------------------------------------
# Atoms input (Issue #152 regression tests)
# ---------------------------------------------------------------------------


class TestQHACalcAtoms:
    """QHA with ASE Atoms input."""

    def test_atoms_with_relaxation(
        self,
        Si_atoms: Atoms,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Atoms input should work when relax_structure=True (default)."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        result = qha_calc.calc(Si_atoms)

        _assert_output_consistency(result)

        ind = result["temperatures"].tolist().index(300)
        assert result["thermal_expansion_coefficients"][ind] == pytest.approx(5.191273165438463e-06, rel=1e-1)

    def test_atoms_without_relaxation(
        self,
        Si_atoms: Atoms,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Atoms input with relax_structure=False must not crash (Issue #152).

        Previously this raised:
            AttributeError: 'Atoms' object has no attribute 'apply_strain'
        """
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            relax_structure=False,
        )
        result = qha_calc.calc(Si_atoms)

        _assert_output_consistency(result)

        # Verify it actually computed something meaningful
        assert all(v > 0 for v in result["volumes"])
        assert len(result["electronic_energies"]) == len(SCALE_FACTORS)

    def test_structure_without_relaxation(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Structure input with relax_structure=False should also work."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            relax_structure=False,
        )
        result = qha_calc.calc(Li2O)

        _assert_output_consistency(result)


# ---------------------------------------------------------------------------
# Temperature / length consistency (Bug #2)
# ---------------------------------------------------------------------------


class TestTemperatureConsistency:
    """Ensure output temperature array matches property arrays."""

    @pytest.mark.parametrize("t_step", [10, 50, 100])
    def test_length_consistency_across_t_steps(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
        t_step: int,
    ) -> None:
        """All output arrays should have consistent lengths for various t_step."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=t_step,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        result = qha_calc.calc(Li2O)

        _assert_output_consistency(result)

    def test_temperatures_are_ascending(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Temperature array should be monotonically increasing."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        result = qha_calc.calc(Li2O)

        temps = result["temperatures"]
        assert all(temps[i] < temps[i + 1] for i in range(len(temps) - 1))

    def test_temperatures_start_near_t_min(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """First temperature should be t_min."""
        t_min = 0
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            t_min=t_min,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        result = qha_calc.calc(Li2O)
        assert result["temperatures"][0] == pytest.approx(t_min)


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------


class TestFileWriting:
    """Test output file writing."""

    # All write_* kwargs and their expected filenames
    WRITE_KWARGS_WITH_FILES = {
        "write_helmholtz_volume": "helmholtz.dat",
        "write_volume_temperature": "volume_temp.dat",
        "write_thermal_expansion": "thermal_expansion.dat",
        "write_gibbs_temperature": "gibbs.dat",
        "write_bulk_modulus_temperature": "bulk_mod.dat",
        "write_heat_capacity_p_numerical": "cp_numerical.dat",
        "write_heat_capacity_p_polyfit": "cp_polyfit.dat",
        "write_gruneisen_temperature": "gruneisen.dat",
    }

    def test_no_files_written_by_default(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
        tmp_path: Path,
    ) -> None:
        """When all write_* are False (default), no files should be created."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        # Run in tmp_path so we can check for unexpected files
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            qha_calc.calc(Li2O)
            created = set(os.listdir(tmp_path))
            assert len(created) == 0, f"Unexpected files: {created}"
        finally:
            os.chdir(original_dir)

    def test_all_files_written_with_custom_paths(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
        tmp_path: Path,
    ) -> None:
        """When all write_* are given paths, all files should be created."""
        write_kwargs = {key: str(tmp_path / fname) for key, fname in self.WRITE_KWARGS_WITH_FILES.items()}

        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
            **write_kwargs,
        )
        qha_calc.calc(Li2O)

        for key, fname in self.WRITE_KWARGS_WITH_FILES.items():
            filepath = tmp_path / fname
            assert filepath.is_file(), f"{key} -> {filepath} was not created"
            assert filepath.stat().st_size > 0, f"{key} -> {filepath} is empty"

    def test_files_written_with_true(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
        tmp_path: Path,
    ) -> None:
        """When write_* is True, default filenames should be used."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            qha_calc = QHACalc(
                calculator=matpes_calculator,
                t_step=T_STEP,
                t_max=T_MAX,
                scale_factors=SCALE_FACTORS,
                fmax=0.1,
                write_helmholtz_volume=True,
                write_gibbs_temperature=True,
            )
            qha_calc.calc(Li2O)

            # Check that default filenames were created
            assert os.path.isfile("helmholtz_volume.dat")
            assert os.path.isfile("gibbs_temperature.dat")

            # Check that others were NOT created
            assert not os.path.isfile("volume_temperature.dat")
            assert not os.path.isfile("thermal_expansion.dat")
        finally:
            os.chdir(original_dir)

    def test_write_count_matches_params(self) -> None:
        """QHACalc should have exactly 8 write_* parameters."""
        import inspect

        params = inspect.signature(QHACalc).parameters
        write_params = [k for k in params if k.startswith("write_")]
        assert len(write_params) == 8


# ---------------------------------------------------------------------------
# Edge cases & parameter validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and parameter handling."""

    def test_dict_input(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Dict input (from previous calc) should work."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        # First run to get a result dict
        result1 = qha_calc.calc(Li2O)

        # Passing the result dict back as input should work
        # (super().calc() extracts final_structure from it)
        result2 = qha_calc.calc(result1)
        _assert_output_consistency(result2)

    def test_few_scale_factors(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Should work with minimum viable number of scale factors (≥4 for EOS fit)."""
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=T_MAX,
            scale_factors=[0.98, 0.99, 1.00, 1.01, 1.02],
            fmax=0.1,
        )
        result = qha_calc.calc(Li2O)
        _assert_output_consistency(result)

    def test_custom_t_range(
        self,
        Li2O: Structure,
        matpes_calculator: PESCalculator,
    ) -> None:
        """Non-default t_min and t_max should work."""
        t_min = 100
        t_max = 500
        qha_calc = QHACalc(
            calculator=matpes_calculator,
            t_step=T_STEP,
            t_max=t_max,
            t_min=t_min,
            scale_factors=SCALE_FACTORS,
            fmax=0.1,
        )
        result = qha_calc.calc(Li2O)
        _assert_output_consistency(result)

        temps = result["temperatures"]
        assert temps[0] >= t_min
        assert temps[-1] <= t_max
