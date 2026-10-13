#!/usr/bin/env bash
# CI helper: remove preinstalled software that the builds do not need from GitHub-hosted Linux runners.
# Usage: bash ci/free_disk.sh [keep_android_sdk]   pass "true" to keep the Android SDK (needed by Android builds)
set -uo pipefail

keep_android="${1:-false}"
echo "Before cleanup:"; df -h /

remove=(
  /usr/share/dotnet
  /opt/ghc
  /usr/local/.ghcup
  /usr/share/swift
  /usr/local/share/boost
  /usr/local/share/powershell
  /usr/local/share/chromium
  /opt/hostedtoolcache/CodeQL
  /opt/hostedtoolcache/go
  /opt/hostedtoolcache/PyPy
  /opt/microsoft/msedge
  /opt/google/chrome
)
if [ "$keep_android" != "true" ]; then
  remove+=(/usr/local/lib/android)
fi
sudo rm -rf "${remove[@]}" || true
sudo docker image prune --all --force > /dev/null 2>&1 || true
sudo apt-get clean || true

echo "After cleanup:"; df -h /

# The runners have 16 GB of RAM and a 4 GB swap file; linking the LTO WebAssembly builds and some EP builds needs
# more, and running out of memory makes the runner shut down. Add a swap file on the (now mostly free) root disk.
if [ ! -e /swapfile-ci ]; then
  { sudo fallocate -l 16G /swapfile-ci && sudo chmod 600 /swapfile-ci && sudo mkswap /swapfile-ci > /dev/null \
    && sudo swapon /swapfile-ci; } || echo "warning: could not add swap"
fi
swapon --show
