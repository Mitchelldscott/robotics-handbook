FROM rust:1.90-slim-bullseye
ARG NODE_VERSION=20

# install dependencies
RUN apt update && apt install -y curl git
RUN apt-get update && apt-get install -y libssl-dev pkg-config build-essential && rm -rf /var/lib/apt/lists/*

# install nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# set env
ENV NVM_DIR=/root/.nvm
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# install node
RUN bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION && nvm use $NODE_VERSION"

# install tools
RUN bash -c "source $NVM_DIR/nvm.sh && npm install -g markdownlint-cli2"
RUN cargo install mdbook mdbook-katex mdbook-bib

# Simple entrypoint: load NVM
WORKDIR /workspace
ENTRYPOINT ["bash", "-c", "source $NVM_DIR/nvm.sh && exec \"$@\"", "--"]
CMD ["/bin/bash"]
