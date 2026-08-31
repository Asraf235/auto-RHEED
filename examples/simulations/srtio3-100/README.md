# SrTiO3 [100] example

This directory contains a small, version-controlled input pair for the
recommended `torch-rheed` dynamical simulation adapter:

- `SrTiO3_100_bulk.txt`: SrTiO3 bulk structure and [100] rocking scan.
- `SrTiO3_100_surf.txt`: the matching surface input.

The bulk input specifies a 10 keV beam, 0° azimuth, a 0.1–7° glancing-angle
scan in 0.01° steps, and a 3.905 Å cubic lattice parameter.

In Auto RHEED, open **Simulation**, select `torch-rheed`, choose these files for
the `bulk` and `surface` inputs, and run the simulation. Runtime results remain
under the ignored `data/simulations/` directory and must not be committed.
