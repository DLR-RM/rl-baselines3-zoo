ARG PARENT_IMAGE
ARG USE_GPU
FROM $PARENT_IMAGE

RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /root/code
ENV VENV /root/venv

RUN \
    mkdir -p ${CODE_DIR}/rl_zoo && \
    pip uninstall -y stable-baselines3 && \
    pip install stable-baselines3[extra] && \
    pip install box2d-py==2.3.5 && \
    pip install pybullet && \
    pip install gym-minigrid && \
    pip install scikit-optimize && \
    pip install optuna && \
    pip install pytablewriter && \
    rm -rf $HOME/.cache/pip

ENV PATH=$VENV/bin:$PATH

COPY docker/entrypoint.sh /tmp/
ENTRYPOINT ["/tmp/entrypoint.sh"]

CMD /bin/bash
