#!/usr/bin/env bash
# build.sh

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p uploads

echo "Build completed successfully!"
