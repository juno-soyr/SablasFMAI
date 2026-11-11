#!/usr/bin/env bash
# Run the whole SABLAS debris-avoidance pipeline from a fresh clone.
#
#   ./run.sh                 full run: checks, smoke test, training (1e6 steps), verification
#   ./run.sh --quick         checks and smoke test only (~1 min)
#   ./run.sh --verify-only   verify an existing checkpoint
#
# Any other argument is passed on to run_sablas.py (see: ./run.sh --help).
# Uses the virtual environment in .venv, creating it if needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=.venv
if [ ! -x "$VENV/bin/python" ]; then
    echo "creating virtual environment in $VENV"
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"

# install only what the pipeline needs (requirements.txt also lists unused solvers)
if ! "$PY" -c "import numpy, matplotlib, torch" 2>/dev/null; then
    echo "installing numpy, matplotlib, torch"
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet numpy matplotlib torch
fi

exec "$PY" run_sablas.py "$@"
