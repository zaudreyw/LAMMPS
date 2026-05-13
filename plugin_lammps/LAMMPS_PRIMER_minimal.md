# LAMMPS Primer (minimal)

LAMMPS (Large-scale Atomic/Massively Parallel Simulator) is an open-source
molecular dynamics code. Tasks require authoring a LAMMPS input script (one
or more `.in` files) that specifies the simulation.

## Where things live (inside the container)

- `/lammps_lib/examples/` — working example input scripts organized by physics
  type (`melt/`, `crack/`, `pour/`, `GCMC/`, `dipole/`, `polymer/`, etc.).
  Treat these as the authoritative reference for command combinations.
- `/lammps_lib/doc/src/` — RST documentation for every command. One `.rst`
  file per command (e.g. `fix_nvt.rst`, `pair_lj_cut.rst`).
- `/workspace/inputs/` — where you must write the final `.in` script(s) and
  any auxiliary files (data files, potential files). Your output goes here.

## Input script structure

LAMMPS commands are **order-sensitive**. The required order is:

```
# 1. Initialization (must come first)
units       <style>           # metal | lj | real | si | cgs | electron | micro | nano
atom_style  <style>           # atomic | charge | molecular | full | bond | angle | …
boundary    p p p             # p=periodic, f=fixed, s=shrink, m=shrink-mapped

# 2. Simulation box + atoms (one of the following)
read_data   <file>            # read pre-built atom positions from a data file
# — OR —
lattice     fcc 3.52          # define lattice (type + scale)
region      box block 0 10 0 10 0 10
create_box  1 box
create_atoms 1 box

# 3. Pair interactions
pair_style  lj/cut 2.5
pair_coeff  * * 1.0 1.0 2.5  # energy, size, cutoff per atom-type pair

# 4. Optional: molecular topology (if atom_style molecular/full)
# bond_style, angle_style, dihedral_style, improper_style, …

# 5. Settings
neighbor    0.3 bin
neigh_modify every 1 delay 0

# 6. Fixes (define integrators, thermostats, barostats, constraints)
fix         1 all nvt temp 300 300 100   # NVT ensemble
# — or nve, npt, etc.

# 7. Output
thermo      100
thermo_style custom step temp press pe ke etotal
dump        1 all atom 1000 dump.lammpstrj

# 8. Run
timestep    0.001
run         10000
```

## Recommended workflow

1. Find a similar existing example via RAG (`mcp__lammps-rag__search_technical`
   for examples; `mcp__lammps-rag__search_navigator` for concepts;
   `mcp__lammps-rag__search_commands` for exact command syntax).
2. `Read` the full example script from `/lammps_lib/examples/...` and adapt
   it to the task's spec.
3. Write the adapted script to `/workspace/inputs/<name>.in` (and any data
   or potential files the script references to `/workspace/inputs/`).
4. Read the file back to verify structure matches the spec.

That's it.

## Common physics patterns

| Physics goal | Key commands |
|---|---|
| NVE (micro-canonical) | `fix N all nve` |
| NVT (canonical) | `fix N all nvt temp T T tau` |
| NPT (iso-baric) | `fix N all npt temp T T tau iso P P ptau` |
| Energy minimization | `minimize etol ftol maxiter maxeval` |
| Grand canonical MC | `fix N all gcmc Nevery Ninsert Nattempt type seed T μ` |
| Rigid bodies | `fix N all rigid/nve molecule` |
| Charged systems | `kspace_style pppm 1e-4` + `pair_style lj/cut/coul/long` |
| Reactive MD (ReaxFF) | `pair_style reaxff NULL` + `pair_coeff * * ffield.reax types…` |

## Units cheat sheet

| `units` | energy | distance | time |
|---|---|---|---|
| `lj` | ε | σ | τ |
| `metal` | eV | Å | ps |
| `real` | kcal/mol | Å | fs |
| `si` | J | m | s |

## Safety

- Never invent command keywords — verify against `/lammps_lib/doc/src/` or
  example scripts when unsure of an argument name.
- `pair_coeff` atom-type indices must match the types defined in `create_box`
  or the data file exactly (1-based).
- If you use `read_data`, write the data file to `/workspace/inputs/` too.
- `dump` filenames are relative to where LAMMPS is run, so use absolute paths
  (e.g. `/workspace/inputs/dump.lammpstrj`) to be safe.
