#! /usr/bin/env bash

SCRIPT_FOLDER="$(cd "$(dirname "${BASH_SOURCE}[0]")" && pwd)"
PROJECT_FOLDER="$(cd "${SCRIPT_FOLDER}" && cd ../.. && pwd)"

SERVER_ENDPOINT="http://localhost:8000"

#######################
# ASR selection
#######################

function select_model() {
    local LANGUAGE="${1}"
    local MODEL_VARIATION="${2}"

    curl -X GET "${SERVER_ENDPOINT}/select/model/${LANGUAGE}/${MODEL_VARIATION}"
}

#######################
# Documents
#######################

function upload_document() {
    local FILEPATH_TO_UPLOAD="${1}"
    curl -X POST \
        -F "file=@${FILEPATH_TO_UPLOAD}" \
        ${SERVER_ENDPOINT}/documents/upload
}

function list_documents() {
    curl -X GET ${SERVER_ENDPOINT}/documents/list
}

function remove_document() {
    local FILENAME_TO_REMOVE="${1}"
    curl -X POST \
        -F "filename=${FILENAME_TO_REMOVE}" \
        ${SERVER_ENDPOINT}/documents/remove
}

#######################
# Embeddings
#######################

function create_embedding() {
    local EMBEDDING_NAME="${1}"
    shift
    local FILENAMES="${1}"
    shift

    for arg in "$@"; do
        FILENAMES+=";${arg}"
    done

    curl -X POST \
        -F "comma_separated_list_filenames=\"${FILENAMES}\"" \
        -F "embedding_name=\"${EMBEDDING_NAME}\"" \
        ${SERVER_ENDPOINT}/embeddings/create
}

function list_embeddings() {
    curl -X GET ${SERVER_ENDPOINT}/embeddings/list
}

function remove_embedding() {
    local EMBEDDING_NAME="${1}"
    curl -X POST \
        -F "embedding_name=\"${EMBEDDING_NAME}\"" \
        ${SERVER_ENDPOINT}/embeddings/remove
}

#######################
# Vector database
#######################

function create_index() {
    local EMBEDDING_NAMES="${1}"
    shift

    for arg in "$@"; do
        EMBEDDING_NAMES+=";${arg}"
    done

    curl -X POST \
        -F "comma_separated_list_embedding_names=\"${EMBEDDING_NAMES}\"" \
        ${SERVER_ENDPOINT}/index/from_embeddings/create
}

function query() {
    local QUERY_STRING="${1}"
    curl -X POST \
        -F "query_string=\"${QUERY_STRING}\"" \
        ${SERVER_ENDPOINT}/query
}

#######################
# Tests
#######################

##### ASR Selection #####

function test_select_model() {
    select_model "pt-br" "large"
}

##### Documents #####

function test_upload_document() {
    upload_document "${PROJECT_FOLDER}/embeddings/documents/hosts.txt"
    upload_document "${PROJECT_FOLDER}/embeddings/documents/piancastagnaio_culture.txt"
    upload_document "${PROJECT_FOLDER}/embeddings/documents/piancastagnaio_history.txt"
}

function test_list_document() {
    list_documents
}

function test_remove_document() {
    local FILENAME_TO_REMOVE="hosts.txt"
    remove_document "${FILENAME_TO_REMOVE}"
}

##### Embeddings #####

function test_create_embedding() {
    create_embedding "my embedding" "hosts.txt" "piancastagnaio_culture.txt"
}

function test_list_embeddings() {
    list_embeddings
}

function test_remove_embedding() {
    remove_embedding "my embedding"
}

##### Vector database #####

function test_create_index() {
    create_index "my embedding"
}

function test_query() {
    query "daniel"
}

##### Full workflow #####

function test_workflow() {
    # test_select_model
    upload_document "${PROJECT_FOLDER}/embeddings/documents/hosts.txt"
    echo ""
    upload_document "${PROJECT_FOLDER}/embeddings/documents/piancastagnaio_culture.txt"
    echo ""
    upload_document "${PROJECT_FOLDER}/embeddings/documents/piancastagnaio_history.txt"
    echo ""
    list_documents
    echo ""

    remove_document "piancastagnaio_history.txt"
    echo ""
    list_documents
    echo ""

    create_embedding "my embedding" "hosts.txt" "piancastagnaio_culture.txt"
    echo ""
    create_embedding "my embedding backup" "hosts.txt" "piancastagnaio_culture.txt"
    echo ""
    list_embeddings
    echo ""
    remove_embedding "my embedding backup"
    echo ""
    list_embeddings
    echo ""

    create_index "my embedding"
    echo ""
    query "piancastagnaio culture"
    echo ""
}

test_workflow
