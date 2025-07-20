#! /usr/bin/env bash

SCRIPT_FOLDER="$(cd "$(dirname "${BASH_SOURCE}[0]")" && pwd)"
PROJECT_FOLDER="$(cd "${SCRIPT_FOLDER}" && cd .. && pwd)"

function create_embedding() {
    local DOC_1="${PROJECT_FOLDER}/documents/hosts.txt"
    local DOC_2="${PROJECT_FOLDER}/documents/piancastagnaio_culture.txt"

    curl -X POST \
        -F "documents_files=@${DOC_1}" \
        -F "documents_files=@${DOC_2}" \
        -F "embedding_name=my-embedding.npy" \
        "http://localhost:8001/create"
}

function get_embedding() {
    curl -X POST \
        -F "text=\"oi tudo bem\"" \
        "http://localhost:8001/get"
}

function list() {
    curl -X GET \
        "http://localhost:8001/list"
}

function remove() {
    curl -X POST \
        -F "embedding_name=my-embedding.npy" \
        "http://localhost:8001/remove"
}

# create_embedding
# list
# remove
get_embedding
