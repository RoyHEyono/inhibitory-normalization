#!/usr/bin/env bash
# Creates a .venv in the project root and installs all dependencies via uv.
# Requires uv: https://docs.astral.sh/uv/getting-started/installation/
set -e

uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

echo "Environment ready. Run: source activate_env.sh"
