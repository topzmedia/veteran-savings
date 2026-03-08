#!/bin/bash
# AI Ad Agency — one-click start
# Usage: bash start.sh  (or double-click in Finder after: chmod +x start.sh)

set -e
cd "$(dirname "$0")"

VENV=".venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
fi

# Activate
source "$VENV/bin/activate"

# Install / upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Start the server
echo ""
echo "Starting AI Ad Agency at http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""
python -m uvicorn ai_ad_agency.web.app:app --reload --port 8000
