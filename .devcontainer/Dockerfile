FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

LABEL maintainer="Vanessasaurus <@vsoch>"

# Match the default user id for a single system so we aren't root
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ARG CLAUDE_CODE_VERSION=latest
ENV USERNAME=${USERNAME}
ENV USER_UID=${USER_UID}
ENV USER_GID=${USER_GID}

# Install extra buildrequires for flux-sched:
RUN sudo apt-get update
RUN sudo apt-get -qq install -y --no-install-recommends \
        libboost-graph-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        libboost-regex-dev \
        python3-yaml \
        libyaml-cpp-dev \
        libedit-dev \
        ninja-build \
        curl \
        nodejs \
        npm \
        iptables \
        ipset \
        iproute2 \
        dnsutils \
        aggregate

# Install Claude
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} && \
    pip install google-adk && \
    pip install --break-system-packages awscli

# Add the group and user that match our ids
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers
USER $USERNAME

 # Copy and set up firewall script
COPY .devcontainer/init-firewall.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "${USERNAME} ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/${USERNAME}-firewall && \
  chmod 0440 /etc/sudoers.d/${USERNAME}-firewall
USER $USERNAME
