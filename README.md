<h1 align="center">
  <img src="https://raw.githubusercontent.com/materialsvirtuallab/matcalc/refs/heads/main/docs/assets/matcalc.png"
width="200" alt="MatCalc" style="vertical-align: middle;" /><br>
</h1>

[![Test](https://github.com/materialsvirtuallab/matcalc/workflows/Test/badge.svg)](https://github.com/materialsvirtuallab/matcalc/workflows/Test/badge.svg)
[![Lint](https://github.com/materialsvirtuallab/matcalc/workflows/Lint/badge.svg)](https://github.com/materialsvirtuallab/matcalc/workflows/Lint/badge.svg)
[![codecov](https://codecov.io/gh/materialsvirtuallab/matcalc/branch/main/graph/badge.svg?token=OR7Z9WWRRC)](https://codecov.io/gh/materialsvirtuallab/matcalc)
[![Requires Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/matcalc?logo=pypi&logoColor=white)](https://pypi.org/project/matcalc?logo=pypi&logoColor=white)
[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matcalc)](https://github.com/materialsvirtuallab/matcalc/blob/main/LICENSE)

## Introduction

MatCalc is a Python library for calculating and benchmarking material properties from the potential energy surface
(PES). The PES can come from DFT or, more commonly, from machine learning interatomic potentials (MLIPs).

Calculating material properties often requires involved setups of various simulation codes. The
goal of MatCalc is to provide a simplified, consistent interface to access these properties with any
parameterization of the PES.

MatCalc is part of the MatML ecosystem, which includes [MatGL] (Materials Graph Library) and [MAML] (MAterials
Machine Learning), the [MatPES] (Materials Potential Energy Surface) dataset, and the [MatCalc] documentation site.

## Documentation

The API documentation and tutorials are available at https://matcalc.ai.

## Installation

```bash
pip install matcalc
```

For MatGL foundation potential support (TensorNet, M3GNet, CHGNet):

```bash
pip install matcalc[matgl]
```

For MAML classical potential support (MTP, GAP, NNP, SNAP):

```bash
pip install matcalc[maml]
```

For benchmarking support:

```bash
pip install matcalc[benchmark]
```

## Architecture

The main base class in MatCalc is `PropCalc` (property calculator). All `PropCalc` subclasses implement a
`calc(pymatgen.Structure | ase.Atoms | dict) -> dict` method that returns a dictionary of properties.

In general, `PropCalc` is initialized with an ML model or [ASE] calculator. The `matcalc.PESCalculator` class
provides convenient access to many foundation potentials (FPs) as well as an interface to MAML for classical
MLIPs such as MTP, NNP, GAP, SNAP, ACE, etc.

### Available Calculators

| Calculator | Description | Key Output Keys |
|---|---|---|
| `RelaxCalc` | Structural relaxation | `energy`, `forces`, `stress`, `a`, `b`, `c`, `alpha`, `beta`, `gamma`, `volume`, `final_structure` |
| `ElasticityCalc` | Elastic constants via strain-stress fitting | `elastic_tensor`, `bulk_modulus_vrh`, `shear_modulus_vrh`, `youngs_modulus`, `residuals_sum` |
| `EOSCalc` | Birch-Murnaghan equation of state | `eos`, `bulk_modulus_bm`, `r2_score_bm` |
| `PhononCalc` | Phonon band structure and thermal properties | `phonon`, `thermal_properties`, `frequencies` |
| `Phonon3Calc` | Third-order force constants and thermal conductivity | `phono3py`, `kappa` |
| `QHACalc` | Quasi-harmonic approximation | `gibbs_temperature`, `thermal_expansion`, `bulk_modulus_temperature` |
| `MDCalc` | Molecular dynamics simulation | `md_structures`, `md_energies` |
| `NEBCalc` | Nudged elastic band / minimum energy path | `mep`, `barrier_energy` |
| `EnergeticsCalc` | Formation and cohesive energies | `formation_energy_per_atom`, `cohesive_energy_per_atom` |
| `SurfaceCalc` | Surface energy calculation | `surface_energy`, `slab_energy`, `final_slab` |
| `AdsorptionCalc` | Adsorption energy on surfaces | `adsorption_energy` |
| `InterfaceCalc` | Coherent interface energy between two bulk structures | `interface_energy` |
| `LAMMPSMDCalc` | LAMMPS-based molecular dynamics | `md_structures`, `md_energies` |
| `ChainedCalc` | Chain multiple PropCalcs in sequence | Combined outputs of all constituent calculators |

### Supported Foundation Potentials

`PESCalculator.load_universal()` — aliased as `matcalc.load_fp()` — supports these models out of the box:

| Model | String Name / Alias | Package |
|---|---|---|
| TensorNet-MatPES-PBE | `"TensorNet-MatPES-PBE-v2025.1-PES"` or `"pbe"` | matgl |
| TensorNet-MatPES-r²SCAN | `"TensorNet-MatPES-r2SCAN-v2025.1-PES"` or `"r2scan"` | matgl |
| M3GNet-MatPES-PBE | `"M3GNet-MatPES-PBE-v2025.1-PES"` or `"m3gnet"` | matgl |
| CHGNet | `"CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"` or `"chgnet"` | matgl |
| MACE-MP | `"MACE"` | mace-torch |
| SevenNet | `"SevenNet"` | sevenn |
| GRACE / TensorPotential | `"GRACE"` or `"TensorPotential"` | tensorpotential |
| ORB | `"ORB"` | orb-models |
| MatterSim | `"MatterSim"` | mattersim |
| FAIRChem | `"FAIRChem"` | fairchem-core |
| PET-MAD | `"PETMAD"` | pet-mad |
| DeePMD | `"DeePMD"` | deepmd-kit |

Aliases are case-insensitive. All pretrained MatGL PES models are auto-discovered if MatGL is installed.

## Basic Usage

MatCalc provides convenient methods to quickly compute properties with minimal code. The following example
computes the elastic constants of Si using the `TensorNet-MatPES-PBE-v2025.1-PES` universal MLIP.

```python
import matcalc as mtc
from pymatgen.ext.matproj import MPRester

mpr = MPRester()
si = mpr.get_structure_by_material_id("mp-149")
c = mtc.ElasticityCalc("TensorNet-MatPES-PBE-v2025.1-PES", relax_structure=True)
props = c.calc(si)
print(f"K_VRH = {props['bulk_modulus_vrh'] * 160.2176621} GPa")
```

The calculated `K_VRH` is about 102 GPa, in reasonably good agreement with the experimental and DFT values.

You can list all supported universal calculators using the `UNIVERSAL_CALCULATORS` enum:

```python
print(mtc.UNIVERSAL_CALCULATORS)
```

MatCalc provides case-insensitive aliases for the recommended PBE and r²SCAN models:

```python
import matcalc as mtc
pbe_calculator = mtc.load_fp("pbe")
r2scan_calculator = mtc.load_fp("r2scan")
```

These currently resolve to the `TensorNet-MatPES-v2025.1` models, but may be updated as better models
become available.

### Parallelization

MatCalc supports trivial parallelization via joblib through the `calc_many` method:

```python
structures = [si] * 20

def serial_calc():
    return [c.calc(s) for s in structures]

def parallel_calc():
    # n_jobs = -1 uses all available processors.
    return list(c.calc_many(structures, n_jobs=-1))

%timeit -n 5 -r 1 serial_calc()
# Output: 8.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 5 loops each)

%timeit -n 5 -r 1 parallel_calc()
# Output: 2.08 s ± 0 ns per loop (mean ± std. dev. of 1 run, 5 loops each)
# This was run on 10 CPUs on a Mac.
```

### Chaining Calculators

`ChainedCalc` runs a sequence of `PropCalc` instances on a structure, accumulating all output properties.
Typically, you start with a `RelaxCalc` followed by property calculators. Set `relax_structure=False` in
downstream calculators to avoid redundant relaxations.

```python
import matcalc as mtc
import numpy as np

calculator = mtc.load_fp("pbe")
relax_calc = mtc.RelaxCalc(
    calculator,
    optimizer="FIRE",
    relax_atoms=True,
    relax_cell=True,
)
energetics_calc = mtc.EnergeticsCalc(
    calculator,
    relax_structure=False  # Skip re-relaxation since we already relaxed above.
)
elast_calc = mtc.ElasticityCalc(
    calculator,
    fmax=0.1,
    norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
    shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
    use_equilibrium=True,
    relax_structure=False,  # Skip re-relaxation since we already relaxed above.
    relax_deformed_structures=True,
)
prop_calc = mtc.ChainedCalc([relax_calc, energetics_calc, elast_calc])
results = prop_calc.calc(structure)
```

`ChainedCalc` also works with `calc_many` for parallel execution over many structures.

### CLI Tool

A command-line interface allows computing properties for any structure file:

```shell
# Compute elastic constants for a CIF file using the default TensorNet model
matcalc calc -p ElasticityCalc -s Li2O.cif

# Use a specific model and write output to a JSON file
matcalc calc -p RelaxCalc -m TensorNet -s structure.cif -o results.json

# Clear the local benchmark data cache
matcalc clear
```

Any format supported by pymatgen's `Structure.from_file()` method is accepted for input structures.
Available property calculators can be checked with `matcalc calc -h`.

## Benchmarking

MatCalc includes a benchmarking framework released alongside [MatPES].

```python
import matcalc as mtc

calculator = mtc.load_fp("TensorNet-MatPES-PBE-v2025.1-PES")
benchmark = mtc.benchmark.ElasticityBenchmark(fmax=0.05, relax_structure=True)
results = benchmark.run(calculator, "TensorNet-MatPES")
```

The entire run takes about 16 minutes when parallelized over 10 CPUs on a Mac.

To run multiple benchmarks on multiple models:

```python
import matcalc as mtc

tensornet = mtc.load_fp("TensorNet-MatPES-PBE-v2025.1-PES")
m3gnet = mtc.load_fp("M3GNet-MatPES-PBE-v2025.1-PES")

elasticity_benchmark = mtc.benchmark.ElasticityBenchmark(fmax=0.5, relax_structure=True)
phonon_benchmark = mtc.benchmark.PhononBenchmark(write_phonon=False)
suite = mtc.benchmark.BenchmarkSuite(benchmarks=[elasticity_benchmark, phonon_benchmark])
results = suite.run({"M3GNet": m3gnet, "TensorNet": tensornet})
results.to_csv("benchmark_results.csv")
```

Full benchmark runs are computationally intensive. Set `n_samples` when initializing a benchmark to test on a
subset before running the full suite. HPC resources are recommended for full benchmark runs.

## Docker Images

Docker images with MatCalc and LAMMPS support are available at the [Materials Virtual Lab Docker Repository].

## Tutorials

Anubhav Jain (@computron) has created a [YouTube tutorial](https://youtu.be/57Elhe4IIhI?si=KbZh5s7HAyNGvmFT) on
using MatCalc to quickly obtain material properties.

## Citing

A manuscript on `matcalc` is currently in the works. In the meantime, please see [`citation.cff`](citation.cff) or the GitHub
sidebar for a BibTeX and APA citation.

[MAML]: https://materialsvirtuallab.github.io/maml/
[MatGL]: https://matgl.ai
[MatPES]: https://matpes.ai
[MatCalc]: https://matcalc.ai
[ASE]: https://wiki.fysik.dtu.dk/ase/
[Materials Virtual Lab Docker Repository]: https://hub.docker.com/orgs/materialsvirtuallab/repositories
