# Sandbox image for "node" entries: Node.js and this repository's bot host, nothing else.
# Build from the repository root: docker build -f docker/agent-node.Dockerfile -t aimoba-agent-node .
FROM node:24-slim
COPY package.json /opt/aimoba/package.json
COPY src/sim /opt/aimoba/src/sim
COPY src/agent /opt/aimoba/src/agent
ENV NODE_NO_WARNINGS=1
USER 65534:65534
WORKDIR /agent
