#!/bin/bash

echo "Setting up AI Avatar Live Environment..."

# Install dependencies
pip install -r requirements.txt

# Create model directories if they don't exist
mkdir -p models/liveportrait
mkdir -p models/rvc_voice

echo "Model directories created."
echo "Please download the following weights manually:"
echo "1. LivePortrait weights -> models/liveportrait/"
echo "2. RVC weight (.pth) and index (.index) -> models/rvc_voice/"

echo "Setup Complete."
