FROM rust:1.90-slim-bullseye
ARG USERNAME=mdyson
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 1. Install dependencies required for Rust and mdbook
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install mdbook and plugins
RUN cargo install mdbook mdbook-mermaid

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# 3. Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip
ENV SHELL /bin/bash

WORKDIR /workspaces/robotics-handbook
CMD ["/bin/bash"]