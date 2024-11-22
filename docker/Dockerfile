# Set up the OS for docker image
FROM ubuntu:latest

# Install the required dependencies and packages (may be possible to optimize)
RUN apt-get update && apt-get install -y git && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
    
# Install Rust and Cargo via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Cargo to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify Cargo installation
RUN cargo --version

# Switch working directory
WORKDIR /app

# Install git repo
ADD https://github.com/PapyPouley/moseiik.git .

# Move to docker branch (only for dockerfile testing)
#RUN git checkout docker

#debug
RUN ls

# Build and run unit tests
CMD cargo test --release

ENTRYPOINT [ "cargo", "test", "--release", "--" ]
