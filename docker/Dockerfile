ARG PARENT_IMAGE
FROM $PARENT_IMAGE

USER root

RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    swig \
    # For building box2d
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

ENV CODE_DIR=/home/$MAMBA_USER/code
COPY requirements.txt /tmp/

RUN \
    eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    mkdir -p ${CODE_DIR}/rl_zoo3 && \
    uv pip uninstall stable-baselines3 && \
    uv pip install --system -r /tmp/requirements.txt && \
    uv pip install --system -e ".[plots,tests]" && \
    uv cache clean

CMD /bin/bash
