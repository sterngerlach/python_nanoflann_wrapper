
# Dockerfile

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ARG CMAKE_VERSION
ARG PYTHON_VERSION

RUN if [[ -z "${CMAKE_VERSION}" ]]; then \
      echo "CMAKE_VERSION variable is not specified"; \
      exit 1; \
      fi
RUN if [[ -z "${PYTHON_VERSION}" ]]; then \
      echo "PYTHON_VERSION variable is not specified"; \
      exit 1; \
      fi

ENV TZ=Asia/Tokyo

# Install essential packages
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
      make \
      build-essential \
      cmake \
      git \
      curl \
      wget \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install CMake
# CMake is installed to /root/${CMAKE_VERSION}/bin/cmake
ENV CMAKE_URL=https://github.com/Kitware/CMake/releases/download
RUN CMAKE_VERSION_NUMBERS=$(echo "${CMAKE_VERSION}" | cut -d"-" -f2) && \
    wget ${CMAKE_URL}/v${CMAKE_VERSION_NUMBERS}/${CMAKE_VERSION}.tar.gz && \
    tar xvfz ${CMAKE_VERSION}.tar.gz && \
    cp -ar ${CMAKE_VERSION} ${HOME}
ENV PATH=${HOME}/${CMAKE_VERSION}/bin:${PATH}

# Install Eigen3
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
      libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for Pyenv
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
      make \
      build-essential \
      libssl-dev \
      zlib1g-dev \
      libbz2-dev \
      libreadline-dev \
      libsqlite3-dev \
      wget \
      llvm \
      libncurses5-dev \
      libncursesw5-dev \
      xz-utils \
      tk-dev \
      libffi-dev \
      liblzma-dev \
      python-openssl \
      git && \
    rm -rf /var/lib/apt/lists/*

# Install Pyenv
RUN git clone --depth=1 https://github.com/pyenv/pyenv /root/.pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

# Install the specified Python
RUN pyenv install $PYTHON_VERSION && \
    pyenv local $PYTHON_VERSION
RUN echo "Python3 location: $(which python3)"
RUN echo "Python3 version: $(python3 --version)"

# RUN PYTHON_VERSION_FULL=$(pyenv install --list | \
#     sort -r --version-sort | \
#     grep -m1 "^ *${PYTHON_VERSION}\.") && \
#     pyenv install $PYTHON_VERSION_FULL && \
#     pyenv local $(pyenv versions | grep ${PYTHON_VERSION})

# Install the prerequisites
RUN pip3 install wheel

# Clone the repository
RUN git clone --depth=1 --recursive \
    https://github.com/sterngerlach/python_nanoflann_wrapper

WORKDIR /python_nanoflann_wrapper

RUN git submodule sync
RUN git submodule update --init --recursive

# Build the Pip package
RUN python3 setup.py build
# Create a wheel
RUN python3 setup.py bdist_wheel
# Copy a wheel
RUN cp dist/*.whl /

WORKDIR /

