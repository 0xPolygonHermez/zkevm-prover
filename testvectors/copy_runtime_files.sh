#!/bin/bash -x

VERSION=v0.3.0.0-rc.1
# Copy the files from the source directory to the destination directory
WORKING_DIR=/mnt/ofs/zkproverc/${VERSION}
SRC=${WORKING_DIR}/basic
DST=.

mkdir -p ${DST}/output
mkdir -p ${DST}/test

cp ${SRC}/zkevm.commit ${DST}/basic.commit
cp ${SRC}/zkevm.const ${DST}/basic.const
cp ${SRC}/zkevm.c12a.const ${DST}/basic.c12a.const
cp ${SRC}/zkevm.c12b.const ${DST}/basic.c12b.const
cp ${SRC}/zkevm.consttree ${DST}/basic.consttree
cp ${SRC}/zkevm.c12a.consttree ${DST}/basic.c12a.consttree
cp ${SRC}/zkevm.c12b.consttree ${DST}/basic.c12b.consttree
cp ${SRC}/zkevm.starkinfo.json ${DST}/basic.starkinfo.json
cp ${SRC}/zkevm.c12a.starkinfo.json ${DST}/basic.c12a.starkinfo.json
cp ${SRC}/zkevm.c12b.starkinfo.json ${DST}/basic.c12b.starkinfo.json
cp ${SRC}/zkevm.verifier_cpp/zkevm.verifier.dat ${DST}/basic.verifier.dat
cp ${SRC}/zkevm.c12a.verifier_cpp/zkevm.c12a.verifier.dat ${DST}/basic.c12a.verifier.dat
cp ${SRC}/zkevm.c12b.verifier_cpp/zkevm.c12b.verifier.dat ${DST}/basic.c12b.verifier.dat
cp ${SRC}/zkevm.c12a.exec ${DST}/basic.c12a.exec
cp ${SRC}/zkevm.c12b.exec ${DST}/basic.c12b.exec

cp ${SRC}/zkevm.g16.verkey.json ${DST}/basic.g16.verkey.json
cp ${SRC}/zkevm.g16.0001.zkey ${DST}/basic.g16.0001.zkey
#cp ${SRC}/rom.json ${DST}/rom.json

#cp ../build/zkProver ${DST}/zkProver
#cp run ${DST}/run
#cp config.json ${DST}/config.json
#cp config_runFile.json ${DST}/config_runFile.json
#cp storage_sm_rom.json ${DST}/storage_sm_rom.json
#cp keccak_script.json ${DST}/keccak_script.json
#cp keccak_connections.json ${DST}/keccak_connections.json
#cp -R tests/* ${DST}/tests