#!/usr/bin/env bash
set -e
sudo apt update
sudo apt -y install git python3 python3-pip python3-venv \
  libatlas-base-dev libjpeg-dev libtiff5-dev libpng-dev \
  libavcodec58 libavformat58 libswscale5 libv4l-0 v4l-utils
# Si usarás Picamera2:
# sudo apt -y install python3-picamera2

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Listo. Configura .env y ejecuta ./run.sh"
