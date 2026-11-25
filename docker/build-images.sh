#!/bin/sh
# Builds the images entries run in. Run from anywhere; rebuild after updating the repository.
set -e
cd "$(dirname "$0")/.."
for kind in node python base; do
  docker build -f "docker/agent-$kind.Dockerfile" -t "aimoba-agent-$kind" .
done
echo "built aimoba-agent-node, aimoba-agent-python, aimoba-agent-base"
