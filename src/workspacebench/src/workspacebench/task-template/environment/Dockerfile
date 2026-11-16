# Workspace-Bench task image. Everything above the workspace step is the same
# for every task, and the workspace step is the same for every task of one
# persona and language, so Docker's layer cache keeps it to 10 workspace layers.
#
# Tooling follows upstream evaluation/docker/Dockerfile (Workspace-Bench @
# 3fbd0f1), minus the Codex / OpenClaw / DeepAgents / DeepSeek harnesses.
FROM node:24-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_PATH=/opt/workspace-bench/office-runtime/node_modules

RUN apt-get update && apt-get install -y --no-install-recommends \
    antiword bash ca-certificates curl fonts-noto-cjk fonts-noto-core ghostscript git graphviz \
    libreoffice-calc libreoffice-common libreoffice-core libreoffice-impress libreoffice-writer \
    pandoc pdftk-java poppler-utils python3 python3-pip python3-venv python-is-python3 qpdf \
    tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng unzip wget zip build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    defusedxml lxml "markitdown[pptx]" numpy openpyxl pandas pdf2image pdfplumber pypdf pypdfium2 \
    Pillow python-docx python-pptx pytesseract reportlab tqdm pyyaml uv remotezip==0.12.6

COPY office-runtime.package.json /opt/workspace-bench/office-runtime/package.json
RUN npm install --prefix /opt/workspace-bench/office-runtime --omit=dev

# Upstream evaluation code at a pinned commit: the output collector and the
# agent-as-a-judge used by the verifier, with its Claude Agent SDK.
ARG WSB_COMMIT=3fbd0f1a136720fece86786545983e26642c3db2
RUN git init -q /opt/wsb \
    && git -C /opt/wsb remote add origin https://github.com/OpenDataBox/Workspace-Bench.git \
    && git -C /opt/wsb sparse-checkout set evaluation/src evaluation/baselines evaluation/skills \
       evaluation/package.json evaluation/package-lock.json \
    && git -C /opt/wsb fetch -q --depth 1 origin "$WSB_COMMIT" \
    && git -C /opt/wsb checkout -q FETCH_HEAD \
    && npm ci --prefix /opt/wsb/evaluation --omit=dev \
    && rm -rf /opt/wsb/.git

# Upstream's office skill pack: for the agent (Harbor's claude-code agent copies
# ~/.claude/skills into its config dir) and for the judge's own config dir.
RUN mkdir -p /root/.claude/skills /opt/wsb/judge-claude/skills \
    && cp -r /opt/wsb/evaluation/skills/office/. /root/.claude/skills/ \
    && cp -r /opt/wsb/evaluation/skills/office/. /opt/wsb/judge-claude/skills/

# The persona workspace, unpacked into /app from the pinned Hugging Face zip.
COPY fetch_workspace.py /opt/wsb/fetch_workspace.py
RUN python3 /opt/wsb/fetch_workspace.py --language {language} --persona {persona_dir} --dest /app \
    && mkdir -p /app/model_output

WORKDIR /app
