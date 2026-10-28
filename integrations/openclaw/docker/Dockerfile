FROM condaforge/miniforge3:24.11.0-0

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    tini \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1000 cellpaint

WORKDIR /opt/CellPainting-Claw
COPY . /opt/CellPainting-Claw

RUN conda env create -f /opt/CellPainting-Claw/environment/cellpainting-claw.environment.yml \
 && conda clean -afy

RUN /opt/conda/envs/cellpainting-claw/bin/pip install -e /opt/CellPainting-Claw[mcp,data-access,test]

USER cellpaint
ENV HOME=/home/cellpaint
ENV PATH=/home/cellpaint/.local/bin:/opt/conda/envs/cellpainting-claw/bin:/opt/conda/bin:$PATH

RUN curl -fsSL https://openclaw.ai/install-cli.sh | bash -s -- --no-onboard

WORKDIR /workspace/state/openclaw/workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/CellPainting-Claw/integrations/openclaw/docker/entrypoint.sh"]
CMD ["gateway"]
