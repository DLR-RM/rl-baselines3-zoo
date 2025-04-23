ARG PARENT_IMAGE=stablebaselines/stable-baselines3
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

# Copy setup file only to install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER ./setup.py ${CODE_DIR}/rl_zoo3/setup.py
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rl_zoo3/version.txt ${CODE_DIR}/rl_zoo3/rl_zoo3/version.txt
COPY --chown=$MAMBA_USER:$MAMBA_USER ./hyperparams ${CODE_DIR}/rl_zoo3/hyperparams
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rl_zoo3/plots ${CODE_DIR}/rl_zoo3/rl_zoo3/plots


RUN \
    eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    cd ${CODE_DIR}/rl_zoo3 && \
    uv pip uninstall stable-baselines3 && \
    uv pip install --system -r /tmp/requirements.txt && \
    uv pip install --system -e ".[plots,tests]" && \
    uv cache clean

CMD /bin/bash
