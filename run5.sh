#!/bin/bash
uv run python scripts/run_experiment.py \
      --run run5 \
      --results-root-dir /home/audrey/repo3/data/eval/run5_results \
      --agents lammps_vanilla lammps_plugin lammps_plugin_validate lammps_plugin_no_hook lammps_plugin_no_rag lammps_plugin_validate_no_rag lammps_deepseek_vanilla lammps_deepseek_plugin lammps_deepseek_plugin_validate lammps_deepseek_no_hook lammps_deepseek_no_rag lammps_deepseek_validate_no_rag \
      --agents-md-path run/AGENTS_lammps.md \
      --plugin-dir plugin_lammps \
      --experiments-dir data/eval/experiments_lammps \
      --lammps-lib-dir /data/shared/lammps_agent_data/data/lammps \
      --vector-db-dir /data/shared/lammps_agent_data/data/vector_db \
      --geos-primer-path plugin_lammps/LAMMPS_PRIMER_minimal.md \
      --strip-baked-primer \
      --ground-truth-dir "" \
      --workers 3
