FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    ca-certificates \
    gnupg \
    libxml2-utils \
    lammps \
    && rm -rf /var/lib/apt/lists/*

# Create /lammps_lib as a documented mount point for the LAMMPS source tree.
# The actual directory is bind-mounted at container run time via
#   -v /path/to/lammps:/lammps_lib:ro
# This mkdir just ensures the path exists so Docker mount semantics work correctly.
RUN mkdir -p /lammps_lib

# Install Node.js 22.x — acpx requires >= 22.12.0
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install cursor-agent
RUN curl -fsSL https://cursor.com/install | bash
ENV PATH="/root/.local/bin:${PATH}"

# Install uv for plugin-provided Python MCP servers. Put it in /usr/local/bin
# so non-root docker users can execute it.
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Claude Code (claude CLI)
RUN npm install -g @anthropic-ai/claude-code

# Install acpx (unified ACP CLI)
RUN npm install -g acpx@latest

WORKDIR /workspace
