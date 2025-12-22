FROM node:20-bookworm-slim

# 1. Install dependencies required for Rust and mdbook
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Rust using rustup (Standard one-line installer)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 3. Add Cargo to the PATH for the rest of the build
ENV PATH="/root/.cargo/bin:${PATH}"

# 4. Install markdownlint-cli2 (No setup needed, Node is already here)
RUN npm install markdownlint-cli2 --global

# 5. Install mdbook and plugins
RUN cargo install mdbook mdbook-mermaid

WORKDIR /workspace
CMD ["/bin/bash"]