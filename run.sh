#!/usr/bin/env bash
set -e
source .venv/bin/activate
[ -f ".env" ] && export $(grep -v '^#' .env | xargs)
python -m croma.main   # o: python lecturaCroma.py
