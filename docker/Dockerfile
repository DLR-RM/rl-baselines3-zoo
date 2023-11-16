ARG PARENT_IMAGE
ARG USE_GPU
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

ENV CODE_DIR /home/$MAMBA_USER/code
COPY requirements.txt /tmp/

RUN \
    eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    mkdir -p ${CODE_DIR}/rl_zoo3 && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install highway-env>=1.7.1 && \
    pip cache purge

CMD /bin/bash
