#!/usr/bin/env bash
set -euo pipefail

minimum_node_major=12

if command -v node >/dev/null 2>&1; then
  node_major="$(node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || true)"
  if [[ "${node_major}" =~ ^[0-9]+$ ]] && (( node_major >= minimum_node_major )); then
    node --version
    exit 0
  fi
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes nodejs
node --version

node_major="$(node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || true)"
if [[ ! "${node_major}" =~ ^[0-9]+$ ]] || (( node_major < minimum_node_major )); then
  echo "The autograder requires Node ${minimum_node_major} or newer." >&2
  exit 1
fi
