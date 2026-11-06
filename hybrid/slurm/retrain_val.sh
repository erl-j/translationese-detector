#!/bin/bash
# Retrain the hybrid models on ALL of trn and score them once on val through the harness.
# CPU only, 1 core. Submit from the hybrid-ml worktree:
#   sbatch hybrid/slurm/retrain_val.sh                 # Phase A + Phase B + val eval
#   sbatch --export=ALL,STEPS=phase_b hybrid/slurm/retrain_val.sh   # after a rule-family merge
#SBATCH -J hybrid-retrain
#SBATCH -p berzelius-hopper-cpu
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -o data/cache/hybrid/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
PY=.venv-hybrid/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
STEPS=${STEPS:-all}
t() { echo "== $(date +%T) $*"; /usr/bin/time -f "   [%U s user, %e s wall, %M KB]" timeout 900 "$@"; }
if [ "$STEPS" = all ]; then
  t $PY -m hybrid.cache --gran 500 100 sent --migrate || true   # reuse old dev.* caches (same bytes as trn)
  t $PY -m hybrid.train
fi
t $PY -m hybrid.engine_cache --gran 500 100 sent
t $PY -m hybrid.stack
if [ "$STEPS" = all ]; then
  t $PY -m hybrid.eval_val --phase-b
else
  t $PY -m hybrid.eval_val --phase-b --only Stacked tsw.engine
fi
t $PY -m hybrid.report
echo "== done $(date +%T)"
