#!/bin/bash
# Quick script to pull latest optimizations and run benchmark

set -e

echo "================================="
echo "Running Optimized Benchmark"
echo "================================="
echo ""

# Pull latest changes
echo "Pulling latest optimizations..."
git fetch origin
git checkout claude/allmos-v2-work-011CV38VASZ67JZG1Wd8xn28
git pull origin claude/allmos-v2-work-011CV38VASZ67JZG1Wd8xn28

echo ""
echo "Latest commit:"
git log -1 --oneline
echo ""

# Run benchmark
echo "Running benchmark..."
python3 bench.py

echo ""
echo "================================="
echo "Benchmark Complete!"
echo "================================="
