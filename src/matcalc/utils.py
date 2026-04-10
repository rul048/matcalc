"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from ase import Atoms
from ase.calculators.calculator import Calculator
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .units import eVA3ToGPa

if TYPE_CHECKING:
    from pathlib import Path

    from maml.apps.pes import LMPStaticCalculator
    from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration
    from pymatgen.core import IMolecule, IStructure


# Listing of supported universal calculators.
# If you update UNIVERSAL_CALCULATORS, you must also update the mapping in
# map_calculators_to_packages in test_utils.py, unless already covered.

_universal_calculators = [
    "M3GNet",
    "CHGNet",
    "MACE",
    "SevenNet",
    "TensorNet",
    "GRACE",
    "TensorPotential",
    "ORB",
    "PBE",
    "r2SCAN",
    "MatterSim",
    "FAIRChem",
    "PETMAD",
    "DeePMD",
]

try:
    # Auto-load all available PES models from matgl if installed.
    import matgl

    _universal_calculators += [
        m for m in matgl.get_available_pretrained_models() if "PES" in m and "ANI-1x-Subset-PES" not in m
    ]
    _universal_calculators = sorted(set(_universal_calculators))
except Exception:  # noqa: BLE001
    warnings.warn("Unable to get pre-trained MatGL universal calculators.", stacklevel=1)

# Provide simple aliases for some common models. The key in MODEL_ALIASES must be lower case.
MODEL_ALIASES = {
    "tensornet": "TensorNet-MatPES-PBE-v2025.1-PES",
    "m3gnet": "M3GNet-MatPES-PBE-v2025.1-PES",
    "chgnet": "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "pbe": "TensorNet-MatPES-PBE-v2025.1-PES",
    "r2scan": "TensorNet-MatPES-r2SCAN-v2025.1-PES",
}


UNIVERSAL_CALCULATORS = Enum("UNIVERSAL_CALCULATORS", {k: k for k in _universal_calculators})  # type: ignore[misc]

# Same strings as enum values; exposed for typing-friendly iteration (e.g. CLI choices).
UNIVERSAL_CALCULATOR_NAMES: tuple[str, ...] = tuple(_universal_calculators)


class PESCalculator(Calculator):
    """
    Class for simulating and calculating potential energy surfaces (PES) using various
    machine learning and classical potentials. It extends the ASE `Calculator` API,
    allowing integration with the ASE framework for molecular dynamics and structure
    optimization.

    PESCalculator provides methods to perform energy, force, and stress calculations
    using potentials such as MTP, GAP, NNP, SNAP, ACE, NequIP, DeePMD and MatGL (M3GNet, TensorNet, CHGNet). The class
    includes utilities to load compatible models for each potential type, making it
    a versatile tool for materials modeling and molecular simulations.

    Attributes:
        potential: MAML LAMMPS static potential backend.
        stress_weight: Factor applied to stress (includes unit conversion).
    """

    implemented_properties = ["energy", "forces", "stress"]  # noqa:RUF012

    def __init__(
        self,
        potential: LMPStaticCalculator,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PESCalculator with a potential from maml.

        Args:
            potential: MAML ``LMPStaticCalculator`` instance.
            stress_unit: ``"GPa"`` or ``"eV/A3"`` for returned stress units.
            stress_weight: Multiplier on stress after unit conversion (default 1.0).
            **kwargs: Forwarded to ``ase.calculators.calculator.Calculator``.
        """
        super().__init__(**kwargs)
        self.potential = potential

        # Handle stress unit conversion
        if stress_unit == "eV/A3":
            conversion_factor = 1 / eVA3ToGPa  # Conversion factor from GPa to eV/A^3
        elif stress_unit == "GPa":
            conversion_factor = 1.0  # No conversion needed if stress is already in GPa
        else:
            raise ValueError(f"Unsupported stress_unit: {stress_unit}. Must be 'GPa' or 'eV/A3'.")

        self.stress_weight = stress_weight * conversion_factor

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """
        Perform calculation for an input Atoms.

        Args:
            atoms: Structure to evaluate.
            properties: ASE property list to compute (defaults to all).
            system_changes: ASE change list; if unchanged, cached results may be reused.
        """
        from ase.calculators.calculator import all_changes, all_properties
        from maml.apps.pes import EnergyForceStress
        from pymatgen.io.ase import AseAtomsAdaptor

        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        structure: Structure | IStructure = AseAtomsAdaptor.get_structure(atoms)  # type: ignore[arg-type,assignment]
        efs_calculator = EnergyForceStress(ff_settings=self.potential)
        energy, forces, stresses = efs_calculator.calculate([structure])[0]

        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": stresses * self.stress_weight,
        }

    @staticmethod
    def load_matgl(path: str | Path, **kwargs: Any) -> Calculator:
        """
        Loads a MATGL model from the specified path and initializes a PESCalculator
        with the loaded model and additional optional parameters.

        This method uses the MATGL library to load a model from the given file path
        or directory. It then configures a calculator using the loaded model and
        the provided keyword arguments.

        Args:
            path: Path to the MatGL model file or pretrained model name.
            **kwargs: Forwarded to the MatGL ASE calculator.

        Returns:
            Configured ASE calculator for the MatGL model.
        """
        import matgl

        model = matgl.load_model(path=path)  # type:ignore[arg-type]
        kwargs.setdefault("stress_unit", "eV/A3")

        if path in (
            "TensorNet-MatPES-PBE-v2025.1-PES",
            "TensorNet-MatPES-r2SCAN-v2025.1-PES",
        ):
            from matgl.ext._ase_pyg import PESCalculator as PESCalculator_pyg

            return PESCalculator_pyg(potential=model, **kwargs)
        from matgl.ext.ase import PESCalculator as PESCalculator_

        return PESCalculator_(potential=model, **kwargs)

    @staticmethod
    def load_mtp(filename: str | Path, elements: list, **kwargs: Any) -> Calculator:
        """
        Load a machine-learned potential (MTPotential) from a configuration file and
        create a calculator object to interface with it.

        This method initializes an instance of MTPotential using a provided
        configuration file and elements. It returns a PESCalculator instance,
        which wraps the initialized potential model.

        Args:
            filename: MTP configuration file path.
            elements: Element symbols for the potential (e.g. ``["Cu"]``).
            **kwargs: Forwarded to ``PESCalculator``.

        Returns:
            ``PESCalculator`` wrapping the MTP model.
        """
        from maml.apps.pes import MTPotential

        model = MTPotential.from_config(filename=filename, elements=elements)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_gap(filename: str | Path, **kwargs: Any) -> Calculator:
        """
        Loads a Gaussian Approximation Potential (GAP) model from the given file and
        returns a corresponding Calculator instance. GAP is a machine learning-based
        potential used for atomistic simulations and requires a specific config file as
        input. Any additional arguments for the calculator can be passed via kwargs,
        allowing customization.

        Args:
            filename: GAP configuration file path.
            **kwargs: Forwarded to ``PESCalculator``.

        Returns:
            ``PESCalculator`` wrapping the GAP model.
        """
        from maml.apps.pes import GAPotential

        model = GAPotential.from_config(filename=str(filename))
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_nnp(
        input_filename: str | Path,
        scaling_filename: str | Path,
        weights_filenames: list,
        **kwargs: Any,
    ) -> Calculator:
        """
        Loads a neural network potential (NNP) from specified configuration files and
        creates a Calculator object configured with the potential. This function allows
        for customizable keyword arguments to modify the behavior of the resulting
        Calculator.

        Args:
            input_filename: NNP input configuration path.
            scaling_filename: NNP scaling parameters path.
            weights_filenames: Paths to NNP weight files.
            **kwargs: Forwarded to ``PESCalculator``.

        Returns:
            ``PESCalculator`` wrapping the NNP model.
        """
        from maml.apps.pes import NNPotential

        model = NNPotential.from_config(
            input_filename=input_filename,
            scaling_filename=scaling_filename,
            weights_filenames=weights_filenames,
        )
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_snap(param_file: str | Path, coeff_file: str | Path, **kwargs: Any) -> Calculator:
        """
        Load a SNAP (Spectral Neighbor Analysis Potential) configuration and create a
        corresponding Calculator instance.

        This static method initializes a SNAPotential instance using the provided
        configuration files and subsequently generates a PESCalculator based on the
        created potential model and additional keyword arguments.

        Args:
            param_file: SNAP parameter file path.
            coeff_file: SNAP coefficient file path.
            **kwargs: Forwarded to ``PESCalculator``.

        Returns:
            ``PESCalculator`` wrapping the SNAP model.
        """
        from maml.apps.pes import SNAPotential

        model = SNAPotential.from_config(param_file=param_file, coeff_file=coeff_file)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_ace(  # pragma: no cover
        basis_set: str | Path | ACEBBasisSet | ACECTildeBasisSet | BBasisConfiguration,
        **kwargs: Any,
    ) -> Calculator:
        """
        Load an ACE (Atomic Cluster Expansion) calculator using the specified basis set.

        This method utilizes the PyACE library to create and initialize a PyACECalculator
        instance with a given basis set. The provided basis set can take various forms including
        file paths, basis set objects, or configurations. Additional customization options
        can be passed through keyword arguments.

        Args:
            basis_set: ACE basis (path, or PyACE basis / configuration object).
            **kwargs: Forwarded to ``PyACECalculator``.

        Returns:
            Initialized PyACE ASE calculator.
        """
        from pyace import PyACECalculator

        return PyACECalculator(basis_set=basis_set, **kwargs)

    @staticmethod
    def load_nequip(  # pragma: no cover
        model_path: str | Path, **kwargs: Any
    ) -> Calculator:
        """
        Loads and returns a NequIP `Calculator` instance from the specified model path.
        This method facilitates the integration of machine learning models into ASE
        by loading a model for atomic-scale simulations.

        Args:
            model_path: Path to the deployed NequIP model.
            **kwargs: Forwarded to ``NequIPCalculator.from_deployed_model``.

        Returns:
            NequIP ASE calculator instance.
        """
        from nequip.ase import NequIPCalculator

        return NequIPCalculator.from_deployed_model(model_path=model_path, **kwargs)

    @staticmethod
    def load_deepmd(  # pragma: no cover
        model_path: str | Path, **kwargs: Any
    ) -> Calculator:
        """
        Loads a Deep Potential Molecular Dynamics (DeePMD) model and returns a `Calculator`
        object for molecular dynamics simulations.

        This method imports the `deepmd.calculator.DP` class and initializes it with the
        given model path and optional keyword arguments. The resulting `Calculator` object
        is used to perform molecular simulations based on the specified DeePMD model.

        The function requires the DeePMD-kit library to be installed to properly import
        and utilize the `DP` class.

        Args:
            model_path: Trained DeePMD model path.
            **kwargs: Forwarded to DeePMD ``DP``.

        Returns:
            DeePMD ASE calculator instance.
        """
        from deepmd.calculator import DP

        return DP(model=model_path, **kwargs)

    @staticmethod
    def load_universal(name: str | Calculator, **kwargs: Any) -> Calculator:  # noqa: C901
        """
        Loads a calculator instance based on the provided name or an existing calculator object. The
        method supports multiple pre-built universal models and aliases for ease of use. If an existing calculator
        object is passed instead of a name, it will directly return that calculator instance. Supported FPs
        include SOTA potentials such as M3GNet, CHGNet, TensorNet, MACE, GRACE, SevenNet, ORB, etc.

        This method is designed to provide a universal interface to load various calculator types, which
        may belong to different domains and packages. It auto-resolves aliases, provides default options
        for certain calculators, and raises errors for unsupported inputs.

        Args:
            name: Model name, alias, or an existing ASE calculator instance.
            **kwargs: Model-specific options passed to the underlying loader.

        Returns:
            ASE calculator instance.

        Raises:
            ValueError: If ``name`` is not a recognized universal model.
        """
        result: Calculator

        if not isinstance(name, str):  # e.g. already an ase Calculator instance
            result = name

        elif any(name.lower().startswith(m) for m in ("m3gnet", "chgnet", "tensornet", "pbe", "r2scan")):
            name = MODEL_ALIASES.get(name.lower(), name)
            result = PESCalculator.load_matgl(name, **kwargs)

        elif name.lower() == "mace":
            from mace.calculators import mace_mp

            result = mace_mp(**kwargs)

        elif name.lower() == "sevennet":
            from sevenn.calculator import SevenNetCalculator

            result = SevenNetCalculator(**kwargs)

        elif name.lower() == "grace" or name.lower() == "tensorpotential":
            from tensorpotential.calculator.foundation_models import grace_fm

            kwargs.setdefault("model", "GRACE-2L-OAM")
            result = grace_fm(**kwargs)

        elif name.lower() == "orb":
            from orb_models.forcefield.calculator import ORBCalculator
            from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS

            model = kwargs.pop("model", "orb-v2")
            device = kwargs.get("device", "cpu")

            orbff = ORB_PRETRAINED_MODELS[model](device=device)
            result = ORBCalculator(orbff, **kwargs)

        elif name.lower() == "mattersim":  # pragma: no cover
            from mattersim.forcefield import MatterSimCalculator

            result = MatterSimCalculator(**kwargs)

        elif name.lower() == "fairchem":  # pragma: no cover
            from fairchem.core import FAIRChemCalculator, pretrained_mlip

            device = kwargs.pop("device", "cpu")
            model = kwargs.pop("model", "uma-s-1")
            task_name = kwargs.pop("task_name", "omat")
            predictor = pretrained_mlip.get_predict_unit(model, device=device)
            result = FAIRChemCalculator(predictor, task_name=task_name, **kwargs)

        elif name.lower() == "petmad":  # pragma: no cover
            from pet_mad.calculator import PETMADCalculator

            result = PETMADCalculator(**kwargs)

        elif name.lower().startswith("deepmd"):  # pragma: no cover
            from pathlib import Path

            from deepmd.calculator import DP

            cwd = Path(__file__).parent.absolute()
            model_path = cwd / "../../tests/pes/DPA3-LAM-2025.3.14-PES" / "2025-03-14-dpa3-openlam.pth"
            model_path = model_path.resolve()
            kwargs.setdefault("model", model_path)
            result = DP(**kwargs)

        else:
            raise ValueError(f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}")

        return result


def to_ase_atoms(structure: Atoms | Structure | Molecule) -> Atoms:
    """
    Converts a given structure into an ASE Atoms object. This function checks
    if the input structure is already an ASE Atoms object. If not, it converts
    a pymatgen Structure object to an ASE Atoms object     using the AseAtomsAdaptor.

    Args:
        structure: ASE ``Atoms``, pymatgen ``Structure``, or ``Molecule``.

    Returns:
        ASE ``Atoms`` for the same system.
    """
    return structure if isinstance(structure, Atoms) else AseAtomsAdaptor.get_atoms(structure)


def to_pmg_structure(structure: Atoms | Structure) -> Structure:
    """
    Converts a given structure of type Atoms or Structure into a Structure
    object. If the input structure is already of type Structure, it is
    returned unchanged. If the input structure is of type Atoms, it is
    converted to a Structure using the AseAtomsAdaptor.

    Args:
        structure: ASE ``Atoms`` or pymatgen ``Structure``.

    Returns:
        Pymatgen ``Structure`` (unchanged if already a structure).
    """
    return structure if isinstance(structure, Structure) else AseAtomsAdaptor.get_structure(structure)  # type: ignore[return-value]


def to_pmg_molecule(structure: Atoms | Structure | Molecule | IMolecule) -> IMolecule:
    """
    Converts a given structure of type Atoms or Structure into a Molecule
    object. If the input structure is already of type Molecule, it is
    returned unchanged. If the input structure is of type Atoms, it is
    converted to a Molecule using the AseAtomsAdaptor.

    Args:
        structure: ASE ``Atoms``, pymatgen ``Structure`` / ``Molecule``, or interface molecule type.

    Returns:
        Pymatgen ``Molecule`` representation.
    """
    if isinstance(structure, Atoms):
        structure = AseAtomsAdaptor.get_molecule(structure)

    return Molecule.from_sites(structure)  # type: ignore[return-value]
