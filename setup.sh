#!/usr/bin/env bash
# Set up a virtual environment with the dependencies needed for kraus_test.ipynb.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON:-python3}"
KERNEL_NAME="${KERNEL_NAME:-kraus-test}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON_BIN' not found on PATH." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib ipympl tqdm qutip jupyterlab ipykernel

python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_NAME"

echo "Environment ready. Activate with 'source $VENV_DIR/bin/activate' then start Jupyter (e.g., 'jupyter lab')."
