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
# Vector database
#######################

function create_index() {
    local INDEX_NAME="${1}"
    curl -X POST \
        -F "index_name=\"${INDEX_NAME}\"" \
        ${SERVER_ENDPOINT}/index/create
}

function remove_index() {
    local INDEX_NAME="${1}"
    curl -X POST \
        -F "index_name=\"${INDEX_NAME}\"" \
        ${SERVER_ENDPOINT}/index/remove
}

function add_document() {
    local INDEX_NAME="${1}"
    local DOCUMENT_NAME="${2}"
    curl -X POST \
        -F "index_name=\"${INDEX_NAME}\"" \
        -F "document_name=\"${DOCUMENT_NAME}\"" \
        ${SERVER_ENDPOINT}/index/add
}

function query() {
    local INDEX_NAME="${1}"
    local QUERY_STRING="${2}"
    curl -X POST \
        -F "index_name=\"${INDEX_NAME}\"" \
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

##### Vector database #####

function test_create_index() {
    create_index "my embedding"
}

function test_add_document() {
    query "my embedding" "hosts"
}

function test_query() {
    query "daniel"
}

##### Full workflow #####

function test_workflow() {
    # test_select_model

    create_index "berry-piancastagnaio"
    echo ""
    upload_document "${PROJECT_FOLDER}/embeddings/documents/hosts.txt"
    echo ""
    upload_document "${PROJECT_FOLDER}/embeddings/documents/piancastagnaio_culture.txt"
    echo ""
    upload_document "${PROJECT_FOLDER}/embeddings/documents/piancastagnaio_history.txt"
    echo ""
    upload_document "${PROJECT_FOLDER}/embeddings/documents/piancastagnaio_monuments.txt"
    echo ""
    add_document "berry-piancastagnaio" "hosts"
    echo ""
    add_document "berry-piancastagnaio" "piancastagnaio_culture"
    echo ""

    query "berry-piancastagnaio" "daniel taste"
    echo ""

    # remove_index "berry-piancastagnaio"
    # echo ""
}

function test_big_document() {
    create_index "berry-jekyll"
    echo ""
    add_document "berry-piancastagnaio" "jekyll_mr_hyde"
    echo ""
    query "berry-piancastagnaio" "scotland yard inspector"
    echo ""

    # remove_index "berry-piancastagnaio"
    # echo ""
}

test_workflow
