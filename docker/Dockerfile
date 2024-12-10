# Set up the OS for docker image
FROM ubuntu:latest

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
    
# Install Rust and Cargo via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Cargo to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Switch working directory
WORKDIR /app

# Install git repo
RUN git clone https://github.com/PapyPouley/moseiik.git .

# Move to docker branch (only for dockerfile testing)
#RUN git checkout UnitTest

# Build and run unit tests
ENTRYPOINT [ "cargo", "test", "--release", "--" ]
