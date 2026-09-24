#!/usr/bin/env sh

CURRENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$CURRENT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -x "$CURRENT_DIR/.venv/bin/python" ]; then
    PYTHON_CMD="$CURRENT_DIR/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD="uv run python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "***** Python not found. Please install dependencies or virtualenv first. *****"
    exit 1
fi

exec $PYTHON_CMD "$CURRENT_DIR/dev.py" "$@"
