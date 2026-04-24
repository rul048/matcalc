---
layout: default
title: Calculating Adsorption Energies.md
nav_exclude: true
---

```python
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from ase.visualize import view
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.surface import SlabGenerator

from matcalc import AdsorptionCalc, PESCalculator, SurfaceCalc
from matcalc.utils import to_ase_atoms

warnings.filterwarnings("ignore", category=UserWarning, module="matgl")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
```


```python
%load_ext autoreload
%autoreload 2
```


```python
# We use the M3GNet potential but it is certainly not a good
# choice as it is not trained for adsorption systems or the most
# accurate for materials available.

calc = PESCalculator.load_universal("M3GNet")
```

    /Users/keith/miniconda3/envs/matcalc-dev/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


In this notebook, we will calculate the minimum adsorption energy of CO on Pt using matcalc and a provided PESCalculator. We will do this in two steps:

1. Determine the lowest surface energy facet using SurfaceCalc
2. Determine the lowest adsorption energy and site using AdsorptionCalc

### 1. Surface Energy


```python
surf_calc = SurfaceCalc(
    calculator=calc,
    relax_bulk=True,
    relax_slab=True,
    fmax=0.01,
    optimizer="BFGS",
    max_steps=200,
    relax_calc_kwargs={},
)
```


```python
%%time
# Build BCC Pt conventional unit cell
bulk = Structure(
    Lattice.cubic(3.924),
    ["Pt"]*4,
    [[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]],
)

# First calculate slab energies for 110 surface, if there are
# multiple terminations aka shifts, there will be multiple slabs returned
miller_index = (1, 1, 0)
slab_results = surf_calc.calc_slabs(
    bulk=bulk,
    miller_index=miller_index,
    min_slab_size=10.0,
    min_vacuum_size=15.0,
    symmetrize=True,
    inplane_supercell=(2, 2),
    slab_gen_kwargs={
        "center_slab": True,
        "max_normal_search": 1,
    },
    get_slabs_kwargs={
        "tol": 0.1,
    },
)
```

    CPU times: user 8.59 s, sys: 43.2 s, total: 51.8 s
    Wall time: 5 s



```python
for slab_result in slab_results:
    for k in slab_result:
        if k not in ["slab", "final_bulk", "final_slab"]:
            print(f"{k}: {slab_result[k]}")
    print("-"*50)
```

    bulk_energy_per_atom: -6.057134628295898
    miller_index: (1, 1, 0)
    slab_energy: -184.0751190185547
    surface_energy: 0.11197308854135114
    --------------------------------------------------


We will now calculate the surface energies of all facets up to a a miller index of 2 to determine the lowest energy site


```python
%%time
# Build BCC Pt conventional unit cell
bulk = Structure(
    Lattice.cubic(3.924),
    ["Pt"]*4,
    [[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]],
)

# You can also use pymatgen's generate_all_slabs to get all unique Miller indices
# up to a certain value, here we get all unique Miller indices up to 2
miller_indices = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 0, 0),
    (2, 1, 0),
    (2, 1, 1),
    (2, 2, 1),
    (2, 2, 0),
]

slab_results = []
for miller_index in miller_indices:
    slab_results += surf_calc.calc_slabs(
        bulk=bulk,
        miller_index=miller_index,
        min_slab_size=10.0,
        min_vacuum_size=15.0,
        symmetrize=True,
        inplane_supercell=(2, 2),
        slab_gen_kwargs={
            "center_slab": True,
            "max_normal_search": 1,
        },
        get_slabs_kwargs={
            "tol": 0.1,
        },
    )
```

    CPU times: user 2min 25s, sys: 9min 14s, total: 11min 40s
    Wall time: 1min 9s



```python
# Some post-processing
df = pd.DataFrame()
for slab_result in slab_results:
    row = {
        "surface_energy": [slab_result["surface_energy"]],
        "miller_index": [slab_result["miller_index"]],
        "shift": [slab_result["slab"].shift],
    }
    df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)

print(df.sort_values("surface_energy").reset_index(drop=True))
```

       surface_energy miller_index     shift
    0        0.084596    (1, 1, 1)  0.166667
    1        0.097773    (2, 2, 1)  0.050000
    2        0.099886    (1, 0, 0)  0.250000
    3        0.099886    (2, 0, 0)  0.250000
    4        0.100212    (2, 1, 1)  0.062500
    5        0.111973    (1, 1, 0)  0.125000
    6        0.111973    (2, 2, 0)  0.125000
    7        0.116478    (2, 1, 0)  0.083333


We see that the Pt(111) surface is lowest energy, so we proceed to calculate adsorption on this surface. Interestingly, M3GNet gets the lowest energy facet correct.

### 2. Adsorption on Pt(111) surface

We start by calculating the adsorption energy when we have the adsorbate, slab and adslab at hand. This method is parallelized in high-throughput calculations.

The structure inputs are quite flexible, they can be ASE or Pymatgen objects. Furthermore you can choose whether or not to relax the adsorbate and/or slab. If you choose not to, you can directly provide the "adsorbate_energy" or "slab_energy_per_atom" but you should still provide the structures. The slab can have a different area but must have the same number of layers, facet and any constraints on the motion of atoms.

Note that it is generally recommended to use experimental or high fidelity values for the adsorbate energies since DFT is not great for molecules. In particular, the O2 molecule triplet state is notoriously challenging fo most DFT functionals. In many cases, oxygen atom/molecule energies are obtained from experiment or linear combinations of reactions. See https://fair-chem.github.io/catalysts/examples_tutorials/OCP-introduction.html. To make matters worse, there is no foundational model known to get both atoms/molecules and materials correct anyway so referencing to a cheap DFT calculation is preferable.


```python
# # Using ASE objects
# slab = build.fcc111("Pt", size=(3, 3, 5), vacuum=15.0)
# # Fix bottom two layers regardles of whether the slab is centered
# c = FixAtoms(indices=[
#     atom.index for atom in slab if atom.position[2] < \
#         np.min(slab.positions, axis=0)[2] + 4.0]
#     )
# slab.set_constraint(c)
# adsorbate = build.molecule("CO")
# adslab = slab.copy()
# build.add_adsorbate(adslab, adsorbate, height=1.8, position="ontop")

# # We can visualize the structures, especially checking
# # that the adsorption site, size and constraints make sense
# view([slab, adsorbate, adslab])
```


```python
# Using pymatgen objects
bulk = Structure(
    Lattice.cubic(3.924),
    ["Pt"]*4,
    [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ],
)
slabgen = SlabGenerator(
    initial_structure=bulk,
    miller_index=(1, 1, 1),
    min_slab_size=10.0,
    min_vacuum_size=15.0,
    center_slab=True,
)
slab = slabgen.get_slabs()[0]
slab.make_supercell((3, 3, 1))

# Apply constraints
minz = np.min(slab.cart_coords, axis=0)[2]
for site in slab:
    if site.coords[2] < minz + 4:
        site.properties["selective_dynamics"] = np.array(
            [False] * 3
        )
    else:
        site.properties["selective_dynamics"] = np.array(
            [True] * 3
        )

adsorbate = Molecule("HHO", [[0, 0, 0], [0, 0, 0.74], [0.76, 0, -0.24]])

adslab = slab.copy()
asf = AdsorbateSiteFinder(slab, height=0.4)
adsite = asf.find_adsorption_sites()["ontop"][0]
adslab = asf.add_adsorbate(adsorbate, adsite)

#Visualize structures
# view([
#     to_ase_atoms(a) for a in [slab, adsorbate, adslab]
# ])
```


```python
ad_calc = AdsorptionCalc(
    calculator=calc,
    relax_slab=True,
    relax_bulk=True,
    relax_adsorbate=False,
    fmax=0.05, #Not a very converged fmax!
    optimizer="BFGS",
    max_steps=200,
    relax_calc_kwargs={},
)
```


```python
results = ad_calc.calc(
    structure={
        "slab": slab,
        "adslab": adslab,
        "adsorbate": adsorbate,
        # "adsorbate_energy": -14.666958808898926, # Optionally provide adsorbate energy
    }
)
```


```python
for key in results:
    if key not in ["slab", "adslab", "adsorbate", "final_adslab", "final_adsorbate", "final_slab"]:
        print(f"{key}: {results[key]}")
```

    adsorbate_energy: -8.20584487915039
    slab_energy: -262.42132568359375
    slab_energy_per_atom: -5.8315850151909725
    adslab_energy: -277.1639099121094
    adsorption_energy: -6.536739349365234


Let's now determine the lowest energy site by going through all the sites and calculating adsorption energies. We will however start by doing a dryrun since this calculation is a bit slow and you should check each of the generated slabs individually to make sure they are sensible!


```python
ac = AdsorptionCalc(
    calculator=calc,
    relax_slab=False,
    relax_bulk=False,
    relax_adsorbate=False,
    fmax=0.1,
    optimizer="BFGS",
    max_steps=200,
    relax_calc_kwargs={},
)
```


```python
bulk = Structure(
    Lattice.cubic(3.924),
    ["Pt"]*4,
    [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ],
)
adsorbate = Molecule("CO", [[0, 0, 0], [0, 0, 1.13]])
```


```python
%%time
results = ac.calc_adslabs(
    adsorbate=adsorbate,
    adsorbate_energy=1.0,
    bulk=bulk,
    miller_index=(1,1,1),
    min_slab_size=10.0,
    min_vacuum_size=10.0,
    inplane_supercell=(2, 2),
    slab_gen_kwargs={},
    get_slabs_kwargs={},
    mi_vec=None,
    # adsorption_sites="all",
    adsorption_sites={"site1": [(0,0,13.5)], "site2": [(2,2,13.5), (0,0,13)]},  #Can specify custom sites
    height=0.7,
    fixed_height=4.0,
    find_adsorption_sites_args={},
    dry_run=True, # Avoids doing any calculations, just generates structures
)
```

    /Users/keith/miniconda3/envs/matcalc-dev/lib/python3.10/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(


    CPU times: user 1.22 s, sys: 1.24 s, total: 2.45 s
    Wall time: 423 ms



```python
adslabs = [to_ase_atoms(res["adslab"]) for res in results]
view(adslabs)
```




    <Popen: returncode: None args: ['/Users/keith/miniconda3/envs/matcalc-dev/bi...>




```python
%%time
results = ac.calc_adslabs(
    adsorbate=adsorbate,
    # adsorbate_energy=1.0, # Optionally provide adsorbate energy to use
    bulk=bulk,
    miller_index=(1,1,1),
    min_slab_size=10.0,
    min_vacuum_size=10.0,
    inplane_supercell=(2, 2),
    slab_gen_kwargs={},
    get_slabs_kwargs={},
    adsorption_sites="all",
    height=0.7,
    fixed_height=4.0,
    find_adsorption_sites_args={},
    dry_run=False,
)
```

    /Users/keith/miniconda3/envs/matcalc-dev/lib/python3.10/site-packages/dgl/core.py:82: DGLWarning: The input graph for the user-defined edge function does not contain valid edges
      dgl_warning(


    CPU times: user 1min 7s, sys: 5min 18s, total: 6min 25s
    Wall time: 37.1 s



```python
df = pd.DataFrame()
for result in results:
    row = {
        "adsorption_energy": [result["adsorption_energy"]],
        "adsorption_site": [result["adsorption_site"]],
        "adsorption_site_index": [result["adsorption_site_index"]],
        "miller_index": [result["miller_index"]],
        "shift": [result["shift"]],
    }
    df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
df.sort_values("adsorption_energy")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adsorption_energy</th>
      <th>adsorption_site</th>
      <th>adsorption_site_index</th>
      <th>miller_index</th>
      <th>shift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-2.160408</td>
      <td>bridge</td>
      <td>0</td>
      <td>(1, 1, 1)</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-2.137474</td>
      <td>ontop</td>
      <td>0</td>
      <td>(1, 1, 1)</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.135719</td>
      <td>hollow</td>
      <td>1</td>
      <td>(1, 1, 1)</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.105980</td>
      <td>hollow</td>
      <td>0</td>
      <td>(1, 1, 1)</td>
      <td>0.166667</td>
    </tr>
  </tbody>
</table>
</div>



From the below reference the PBE adsorption energies are:

- Top: -1.80eV
- Bridge: -1.87eV
- Hollow: -1.88eV

So we're about 0.2eV off which isn't bad considering we used the M3GNet model with very loose convergence criteria. The hollow site is indeed the most stable.

Reference: https://pubs.acs.org/doi/full/10.1021/acscatal.8b00214
