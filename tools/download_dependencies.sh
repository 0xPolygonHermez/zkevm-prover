#!/bin/bash

# This script downloads the dependencies for prover
DOWNLOAD_URL=http://de012a78750e59b808d922b39535e862.s3-website-eu-west-1.amazonaws.com/v0.4.0.0-rc.1
DOWNLOAD_DIR=./testvectors/

# Download the dependencies
wget ${DOWNLOAD_URL}/zkevm.const -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12a.const -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12b.const -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.consttree -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12a.consttree -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12b.consttree -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.starkinfo.json -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12a.starkinfo.json -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12b.starkinfo.json -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.verifier.dat -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12a.verifier.dat -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12b.verifier.dat -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12a.exec -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.c12b.exec -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.g16.verkey.json -P ${DOWNLOAD_DIR}
wget ${DOWNLOAD_URL}/zkevm.g16.0001.zkey -P ${DOWNLOAD_DIR}

