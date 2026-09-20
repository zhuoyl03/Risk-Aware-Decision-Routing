#!/usr/bin/env bash
# Run from any directory. Use RADR_PYTHON=/path/to/python when Conda is inactive.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
RADR_PYTHON="${RADR_PYTHON:-python}"
mode="${1:---cached}"
if [[ "$mode" != "--cached" && "$mode" != "--full" ]]; then
    echo "Usage: bash scripts/reproduce.sh [--cached|--full]" >&2
    exit 2
fi
mkdir -p out
"$RADR_PYTHON" -m unittest discover -s tests -v 2>&1 | tee out/tests.log
if [[ "$mode" == "--full" ]]; then
    "$RADR_PYTHON" -u -m radr.extract 2>&1 | tee out/extraction.log
fi
"$RADR_PYTHON" -u -m radr.experiment 2>&1 | tee out/experiment.log
if [[ "$mode" == "--full" ]]; then
    "$RADR_PYTHON" -u -m radr.stress 2>&1 | tee out/stress.log
fi
"$RADR_PYTHON" -m radr.report
"$RADR_PYTHON" -m radr.predict --feature-cache out/features/test.npz --limit 20 --output out/demo.json
