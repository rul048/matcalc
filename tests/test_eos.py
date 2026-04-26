"""Tests for EOSCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ase.filters import ExpCellFilter

from matcalc import EOSCalc

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_eos_calc(
    Li2O: Structure,
    LiFePO4: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for EOSCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    eos_calc = EOSCalc(matpes_calculator, fmax=0.1, relax_calc_kwargs={"cell_filter": ExpCellFilter})
    result = eos_calc.calc(Li2O)

    assert result["bulk_modulus_bm"] == pytest.approx(80.24729086994387, rel=1e-1)
    assert {*result["eos"]} == {"volumes", "energies"}
    assert result["eos"]["volumes"] == pytest.approx(
        [
            18.428878249959727,
            19.684974412489215,
            20.99688808280458,
            22.365832684988153,
            23.793021643122202,
            25.279668381289053,
            26.826986323571,
            28.436188894050332,
            30.108489516809357,
            31.84510161593039,
            33.64723861549573,
        ],
        rel=1e-1,
    )
    assert result["eos"]["energies"] == pytest.approx(
        [
            -13.577718734741211,
            -13.87842845916748,
            -14.080720901489258,
            -14.209554672241211,
            -14.278072357177734,
            -14.29671573638916,
            -14.274452209472656,
            -14.218313217163086,
            -14.133645057678223,
            -14.024273872375488,
            -13.892631530761719,
        ],
        rel=1e-3,
    )
    eos_calc = EOSCalc(matpes_calculator, relax_structure=False)
    results = list(eos_calc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus_bm"] == pytest.approx(69.10044665311513, rel=1e-1)


def test_eos_calc_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for EOSCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    eos_calc = EOSCalc(matpes_calculator, fmax=0.1, relax_structure=False)
    result = eos_calc.calc(Si_atoms)

    assert result["bulk_modulus_bm"] == pytest.approx(76.57813407426684, rel=1e-1)
