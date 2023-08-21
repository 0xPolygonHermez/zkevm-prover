#!/bin/bash
set -e

echo "Please ensure that you run this script from the root of the zkevm-prover repository."
echo "Press Enter to continue..."
read

ARCHIVE_NAME="v1.1.0-rc.1-fork.4"
ARCHIVE_EXTENSION=".tgz"
ARCHIVE_URL="https://de012a78750e59b808d922b39535e862.s3.eu-west-1.amazonaws.com/${ARCHIVE_NAME}${ARCHIVE_EXTENSION}"

wget -c ${ARCHIVE_URL}
tar -xzvf ${ARCHIVE_NAME}${ARCHIVE_EXTENSION}
rm -rf config
cp -R ${ARCHIVE_NAME}/config .