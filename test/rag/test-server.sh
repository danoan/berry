#! /usr/bin/bash

function create_embedding_from_documents() {
    local D1="/home/daniel/Projects/vosk-asr/embeddings/my_embedding.npy"
    local D2="/home/daniel/Projects/vosk-asr/embeddings/my_embedding.npy"

    curl -X POST \
        -F "embedding_files=@${D1}" \
        -F "embedding_files=@${D2}" \
        http://localhost:8002/from_embeddings/create
}

function list_indexes() {
    curl -X GET \
        http://localhost:8002/list
}

function query() {
    curl -X POST \
        -F "query_string=\"oi tudo bem\"" \
        http://localhost:8002/query
}

# create_embedding_from_documents
# list_indexes
query
