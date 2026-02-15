#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "==> Installing rehab-os with dev dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

echo "==> Verifying imports..."
python -c "from rehab_os.config import get_settings; print('✓ rehab_os imports OK')"

echo ""
echo "Done! Activate with:  source venv/bin/activate"
