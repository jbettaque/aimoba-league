# Sandbox image for "command" entries: a plain Debian system. The entry brings everything it needs.
# Build from the repository root: docker build -f docker/agent-base.Dockerfile -t aimoba-agent-base .
FROM debian:bookworm-slim
USER 65534:65534
WORKDIR /agent
