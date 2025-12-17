FROM python:3.12
ARG TARGETPLATFORM
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ADD https://deb.nodesource.com/setup_23.x /tmp/nodesource_setup.sh
RUN bash /tmp/nodesource_setup.sh
RUN apt-get install -y nodejs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT=/usr/local
WORKDIR /config
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=/config/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/config/pyproject.toml \
    uv sync --locked --no-editable --no-install-project
WORKDIR /app
