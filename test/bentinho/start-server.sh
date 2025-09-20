#! /usr/bin/env bash

SCRIPT_FOLDER="$(cd "$(dirname "${BASH_SOURCE}[0]")" && pwd)"
PROJECT_FOLDER="$(cd "${SCRIPT_FOLDER}" && cd .. && pwd)"

pushd "${PROJECT_FOLDER}" >/dev/null
(
    PYTHON_PATH="." uvicorn --port 8000 bentinho.bentinho:app
)
popd >/dev/null
