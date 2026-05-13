# LAMMPS Overview

## What is LAMMPS?

LAMMPS (Large-scale Atomic/Massively Parallel Simulator) is an open-source
molecular dynamics (MD) code maintained by Sandia National Laboratories.  It
simulates how collections of atoms or molecules move over time by integrating
Newton's laws of motion.  Given an initial geometry and a set of inter-atomic
forces, LAMMPS steps the system forward in time (typically femtoseconds to
nanoseconds) and records positions, velocities, energies, and stresses.

LAMMPS is domain-general: the same code handles metals, polymers, proteins,
granular materials, plasma, and coarse-grained models.  What changes between
simulations is the **force field** (how atoms attract and repel each other)
and the **ensemble** (what thermodynamic quantities are held constant).

---

## How molecular dynamics works

1. **Initial conditions** — atoms are placed in a box (lattice, random, or
   read from a file) and assigned velocities drawn from a Maxwell-Boltzmann
   distribution at the target temperature.
2. **Force calculation** — for every timestep, LAMMPS computes the force on
   each atom from the chosen potential (pair, bond, angle, long-range, etc.).
3. **Integration** — positions and velocities are updated via a numerical
   integrator (Verlet algorithm by default).  Each timestep is typically
   0.5–2 fs for molecular systems or 0.001–0.005 τ in reduced LJ units.
4. **Output** — thermodynamic data (energy, temperature, pressure) is written
   every N steps; atom positions are written to a dump file for visualization.

A typical production run has three phases:

| Phase | Purpose | Typical length |
|---|---|---|
| Energy minimization | Remove bad contacts from the initial geometry | 100–1000 iterations |
| Equilibration | Bring temperature/pressure to target values | 10³–10⁵ steps |
| Production | Collect statistics | 10⁵–10⁷ steps |

---

## The three evaluation tasks

These three tasks were chosen to cover a range of complexity and physics types.
They are not the only things LAMMPS can do — they are representative benchmarks.

### `lj_melt` — Lennard-Jones melt

**What it simulates.** Argon atoms starting in a perfect FCC crystal lattice,
heated above the melting point so the crystal disorders into a liquid.

**Why it matters.** The Lennard-Jones (LJ) potential is the simplest realistic
pair interaction: a repulsive core and an attractive well, described by two
parameters (ε, σ).  The LJ melt is the "Hello World" of MD — it has been run
billions of times, its equilibrium properties are tabulated, and it serves as
a universal correctness check for new codes and new configurations.

**Key physics parameters.**
- `units lj` — dimensionless reduced units where ε = σ = m = 1
- Density ρ = 0.8442 σ⁻³ (liquid phase at T* = 1.44)
- Cutoff 2.5σ — standard for LJ; truncates the potential where it is small
- NVE then NVT — brief microcanonical run to relax forces, then canonical run
  to sample the liquid ensemble

**What correct output looks like.** Temperature should stabilize near 1.44,
potential energy per atom near −6 ε, and the dump file should show a
disordered liquid structure.

---

### `crack_2d` — 2D crack propagation

**What it simulates.** A two-dimensional sheet of atoms with a pre-cut notch
(a thin slot from the left edge to the center).  The top and bottom edges are
pulled apart at a constant velocity, loading the crack tip in Mode I (opening
mode) fracture.  The simulation shows the crack advancing, atom by atom.

**Why it matters.** Atomistic fracture mechanics reveals phenomena invisible
to continuum models: crack tip emission of dislocations, bond-breaking
sequences, and the relationship between atomic-scale structure and macroscopic
toughness.  The 2D LJ crack is the standard pedagogical example.

**Key physics parameters.**
- `dimension 2` + `boundary p s p` — 2D simulation, free surfaces in y
- Square LJ lattice at spacing 1.0σ (solid, below melting point)
- Notch deletes atoms in a thin rectangle at mid-height, left half of box
- `fix move` on top/bottom strips applies a constant displacement rate
- NVE dynamics — no thermostat; kinetic energy released at crack tip heats
  the system locally, which is physically correct

**What correct output looks like.** Crack should advance from the notch tip
toward the right.  Dump file stress components should show stress concentration
at the crack tip.  Temperature rises as the crack propagates (energy release).

---

### `nvt_water` — SPC/E water NVT

**What it simulates.** 216 liquid water molecules at 300 K in a periodic cubic
box, using the SPC/E (Extended Simple Point Charge) force field.

**Why it matters.** Water is the most important solvent in chemistry and
biology.  SPC/E is the workhorse water model: three-point (one O, two H),
with fixed partial charges and rigid geometry maintained by SHAKE constraints.
It reproduces density, diffusion coefficient, and radial distribution functions
of liquid water well.

**Key physics parameters.**
- `units real` — kcal/mol, Å, fs (the standard for biomolecular simulations)
- `atom_style full` — every atom carries a molecule ID, type, charge, and
  coordinates; required for bonded + charged systems
- `pair_style lj/cut/coul/long` + `kspace_style pppm` — short-range LJ plus
  long-range electrostatics via Particle-Particle Particle-Mesh (PPPM).
  Electrostatics cannot be truncated at a short cutoff without large errors.
- `fix shake` — constrains O-H bonds and H-O-H angle to their equilibrium
  values, allowing a 2 fs timestep instead of ~0.5 fs
- `minimize` before `run` — removes any bad contacts in the initial geometry

**What correct output looks like.** Temperature should fluctuate around 300 K.
Total energy should be stable.  Potential energy per molecule ~ −10 kcal/mol.
If run long enough, O-O radial distribution function should show a first peak
near 2.8 Å.

---

## Key simulation concepts

### Ensembles

The ensemble defines what is held constant during the simulation.

| Ensemble | Fixed quantities | LAMMPS fix | Use case |
|---|---|---|---|
| NVE | N atoms, Volume, Energy | `fix nve` | Isolated system, testing |
| NVT | N atoms, Volume, Temperature | `fix nvt` | Liquid properties, most production |
| NPT | N atoms, Pressure, Temperature | `fix npt` | Phase transitions, density calculation |
| GCMC | μ (chemical potential), V, T | `fix gcmc` | Adsorption, open systems |

### Force fields

| Force field | Best for | LAMMPS pair_style |
|---|---|---|
| Lennard-Jones | Noble gases, coarse-grained | `lj/cut` |
| EAM | Metals (Cu, Al, Fe, …) | `eam`, `eam/alloy` |
| Tersoff / Stillinger-Weber | Covalent solids (Si, C, …) | `tersoff`, `sw` |
| CHARMM / AMBER / OPLS | Biomolecules, polymers | `lj/cut/coul/long` |
| ReaxFF | Reactive chemistry (bond breaking) | `reaxff` |
| SPC/E, TIP4P | Water | `lj/cut/coul/long` + `kspace` |

### Timestep guidelines

Too large → energy drift, simulation blows up.  Too small → wasted compute.

| System | Typical timestep |
|---|---|
| LJ reduced units | 0.001–0.005 τ |
| Metal (EAM) | 1–2 fs |
| Molecular (real units, rigid bonds) | 1–2 fs |
| Molecular (real units, flexible bonds) | 0.5–1 fs |
| Coarse-grained | 10–50 fs |

### What makes a simulation "correct"

A correct LAMMPS script must satisfy four levels:

1. **Syntactic** — parses without error; all command keywords are valid.
2. **Structural** — commands appear in the right order; required commands for
   the chosen atom_style and pair_style are all present.
3. **Physical** — parameters (temperature, density, cutoff, timestep, charges)
   match the target system and are physically reasonable.
4. **Convergent** — the system equilibrates; energy and temperature reach
   stable plateaus rather than drifting or exploding.

Levels 1–3 can be checked automatically.  Level 4 requires either running the
simulation or an experienced eye on the thermo output.

---

## Our evaluation setup

### Tasks

Three tasks covering the complexity spectrum:

| Task | Ensemble | Force field | Complexity |
|---|---|---|---|
| `lj_melt` | NVT | LJ pair only | Low |
| `crack_2d` | NVE + driven boundaries | LJ pair only | Medium |
| `nvt_water` | NVT | LJ + Coulomb + PPPM + bonds/angles + SHAKE | High |

`lj_melt_minimal` is a stripped-down version of `lj_melt` with only three
lines of instruction (vs. the full step-by-step spec) to test how much the
agent relies on explicit guidance vs. its own physics knowledge.

### Agent variants

| Agent | RAG (doc search) | Stop hook (validation) |
|---|---|---|
| `lammps_vanilla` | ✗ | ✗ |
| `lammps_plugin_no_rag` | ✗ | ✓ |
| `lammps_plugin` | ✓ | ✓ |
| `lammps_plugin_validate` | ✓ | ✓ + mid-task validate tool |

### Scoring

Each task is scored on two dimensions:

- **Structural score (0–1)** — automated checks: correct `units`, `atom_style`,
  integrator fix, kspace for charged systems, `dimension 2` for crack, etc.
  A score of 1.0 means every required command is present with the right value.

- **LLM judge score (0–10)** — an LLM reads the task spec, the ground truth
  script, and the agent script side by side.  It assigns a score and a short
  rationale.  Rubric: 10 = perfect match; 7–9 = minor parameter deviations;
  4–6 = correct approach, missing components; 1–3 = wrong ensemble or units;
  0 = no valid script.

### Ground truth

Reference `.in` scripts that exactly implement each task's specification live
at `data/eval/experiments_lammps_gt/<task>/inputs/`.  They are used by the
LLM judge for comparison and by the evaluator's structural parser to confirm
the agent used the same physics approach.  The water task also includes a
pre-generated `water.data` geometry file (216 SPC/E molecules, 6×6×6 cubic
grid at 3.1 Å spacing) and the Python script used to generate it.
