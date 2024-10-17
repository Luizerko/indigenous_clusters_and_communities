#!/bin/bash

# Base URL of the collection and base data directory
BASE_URL="http://tainacan.museudoindio.gov.br/wp-json/tainacan/v2/collection/471/items/?perpage=96&order=ASC&orderby=date&exposer=json-flat&paged="
OUTPUT_DIR="./data/"

# Loop through whole collection (219 pages for 20965 items)
for page in {1..219}
do

    # Constructing URL, choosing output file name and downloading JSON
    URL="${BASE_URL}${page}"
    OUTPUT_FILE="${OUTPUT_DIR}/page_${page}.json"
    echo "Downloading page ${page}..."
    curl -s "$URL" -o "$OUTPUT_FILE"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Page ${page} saved to ${OUTPUT_FILE}"
    else
        echo "Failed to download page ${page}"
    fi
done
