FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

ARG USERNAME=dysedev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user and grant sudo privileges
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ==================== Install Dependencies ====================
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    software-properties-common \
    unzip \
    git \
    vim \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade -y

# ==================== Install Python 3.13 ====================
RUN apt-get update && apt-get install -y python3-dev python3-pip

# ==================== Install Julia ====================
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz \
  && tar -xzf julia-1.10.5-linux-x86_64.tar.gz \
  && mv julia-1.10.5 /opt/julia \
  && ln -s /opt/julia/bin/julia /usr/local/bin/julia \
  && rm julia-1.10.5-linux-x86_64.tar.gz

USER ${USERNAME}

RUN julia -e 'using Pkg; Pkg.add(["Pluto", "ControlSystems", "LinearAlgebra"]); Pkg.precompile()'

# ==================== Install Rust 1.90 ====================
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.90.0
ENV PATH="/home/${USERNAME}/.cargo/bin:${PATH}"
RUN rustup update

# ==================== Install mdBook ====================
RUN cargo install mdbook mdbook-mermaid


WORKDIR /workspaces/robotics-handbook
CMD ["/bin/bash"]