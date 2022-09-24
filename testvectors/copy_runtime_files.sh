#!/bin/bash -x

VERSION=v0.3.0.0-rc.1
# Copy the files from the source directory to the destination directory
WORKING_DIR=/mnt/ofs/zkproverc/${VERSION}
SRC=${WORKING_DIR}/n23
DST=${WORKING_DIR}/runtime

mkdir -p ${DST}/output
mkdir -p ${DST}/test

cp ${SRC}/zkevm.const ${DST}/zkevm.const
cp ${SRC}/zkevm.c12a.const ${DST}/zkevm.c12a.const
cp ${SRC}/zkevm.c12b.const ${DST}/zkevm.c12b.const
cp ${SRC}/zkevm.consttree ${DST}/zkevm.consttree
cp ${SRC}/zkevm.c12a.consttree ${DST}/zkevm.c12a.consttree
cp ${SRC}/zkevm.c12b.consttree ${DST}/zkevm.c12b.consttree
cp ${SRC}/zkevm.starkinfo.json ${DST}/zkevm.starkinfo.json
cp ${SRC}/zkevm.c12a.starkinfo.json ${DST}/zkevm.c12a.starkinfo.json
cp ${SRC}/zkevm.c12b.starkinfo.json ${DST}/zkevm.c12b.starkinfo.json
cp ${SRC}/zkevm.verifier_cpp/zkevm.verifier.dat ${DST}/zkevm.verifier.dat
cp ${SRC}/zkevm.c12a.verifier_cpp/zkevm.c12a.verifier.dat ${DST}/zkevm.c12a.verifier.dat
cp ${SRC}/zkevm.c12b.verifier_cpp/zkevm.c12b.verifier.dat ${DST}/zkevm.c12b.verifier.dat
cp ${SRC}/zkevm.c12a.exec ${DST}/zkevm.c12a.exec
cp ${SRC}/zkevm.c12b.exec ${DST}/zkevm.c12b.exec

cp ${SRC}/zkevm.g16.verkey.json ${DST}/zkevm.g16.verkey.json
cp ${SRC}/zkevm.g16.0001.zkey ${DST}/zkevm.g16.0001.zkey
cp ${SRC}/rom.json ${DST}/rom.json

cp ../build/zkProver ${DST}/zkProver
cp run ${DST}/run
cp config.json ${DST}/config.json
cp config_runFile.json ${DST}/config_runFile.json
cp storage_sm_rom.json ${DST}/storage_sm_rom.json
cp keccak_script.json ${DST}/keccak_script.json
cp keccak_connections.json ${DST}/keccak_connections.json