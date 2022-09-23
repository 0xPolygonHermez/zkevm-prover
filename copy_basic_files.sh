#!/bin/bash -x
VERSION=v0.3.0.0-rc.1
# Copy the files from the source directory to the destination directory
WORKING_DIR=/home/edu/zkevm-prover
SRC=/mnt/ofs/zkproverc/${VERSION}/basic
DST=${WORKING_DIR}/testvectors

cp ${SRC}/zkevm.commit ${DST}/basic.commit
cp ${SRC}/zkevm.const ${DST}/basic.const
cp ${SRC}/zkevm.consttree ${DST}/basic.consttree
cp ${SRC}/zkevm.starkinfo.json ${DST}/basic.starkinfo.json
cp ${SRC}/zkevm.c12a.const ${DST}/basic.c12a.const
cp ${SRC}/zkevm.c12a.consttree ${DST}/basic.c12a.consttree
cp ${SRC}/zkevm.c12a.exec ${DST}/basic.c12a.exec
cp ${SRC}/zkevm.c12a.starkinfo.json ${DST}/basic.c12a.starkinfo.json
cp ${SRC}/zkevm.verifier_cpp/zkevm.verifier.dat ${DST}/basic.verifier.dat
cp ${SRC}/zkevm.c12a.verifier_cpp/zkevm.c12a.verifier.dat ${DST}/basic.c12a.verifier.dat
cp ${SRC}/zkevm.c12a.verifier_cpp/zkevm.c12a.verifier.cpp ${DST}/../src/starkpil/test/zkevm_c12a_verifier_cpp/zkevm.c12a.verifier.cpp

cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step2.cpp ${DST}/../src/starkpil/test/basic.c12b.chelpers.step2.cpp
cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step3.cpp ${DST}/../src/starkpil/test/basic.c12b.chelpers.step3.cpp
cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step4.cpp ${DST}/../src/starkpil/test/basic.c12b.chelpers.step4.cpp
cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step42ns.cpp ${DST}/../src/starkpil/test/basic.c12b.chelpers.step42ns.cpp
cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step52ns.cpp ${DST}/../src/starkpil/test/basic.c12b.chelpers.step52ns.cpp

cp ${SRC}/zkevm.g16.0001.zkey ${DST}/basic.g16.0001.zkey
cp ${SRC}/zkevm.c12b.const ${DST}/basic.c12b.const
cp ${SRC}/zkevm.c12b.consttree ${DST}/basic.c12b.consttree
cp ${SRC}/zkevm.c12b.starkinfo.json ${DST}/basic.c12b.starkinfo.json
cp ${SRC}/zkevm.c12b.exec ${DST}/basic.c12b.exec
cp ${SRC}/zkevm.c12b.verifier_cpp/zkevm.c12b.verifier.dat ${DST}/basic.c12b.verifier.dat
cp ${SRC}/zkevm.g16.verkey.json ${DST}/basic.g16.verkey.json