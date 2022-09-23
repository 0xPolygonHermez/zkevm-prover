#!/bin/bash -x
VERSION=v0.3.0.0-rc.1
# Copy the files from the source directory to the destination directory
WORKING_DIR=/home/edu/zkevm-prover
SRC=/mnt/ofs/zkproverc/${VERSION}/n23
DST=${WORKING_DIR}/testvectors

cp ${SRC}/zkevm.commit ${DST}/zkevm.commit
cp ${SRC}/zkevm.const ${DST}/zkevm.const
cp ${SRC}/zkevm.consttree ${DST}/zkevm.consttree
cp ${SRC}/zkevm.starkinfo.json ${DST}/zkevm.starkinfo.json
cp ${SRC}/zkevm.c12a.const ${DST}/zkevm.c12a.const
cp ${SRC}/zkevm.c12b.const ${DST}/zkevm.c12b.const
cp ${SRC}/zkevm.c12a.consttree ${DST}/zkevm.c12a.consttree
cp ${SRC}/zkevm.c12b.consttree ${DST}/zkevm.c12b.consttree
cp ${SRC}/zkevm.c12a.exec ${DST}/zkevm.c12a.exec
cp ${SRC}/zkevm.c12b.exec ${DST}/zkevm.c12b.exec
cp ${SRC}/zkevm.c12a.starkinfo.json ${DST}/zkevm.c12a.starkinfo.json
cp ${SRC}/zkevm.c12b.starkinfo.json ${DST}/zkevm.c12b.starkinfo.json
cp ${SRC}/zkevm.verifier_cpp/zkevm.verifier.dat ${DST}/zkevm.verifier.dat
cp ${SRC}/zkevm.c12a.verifier_cpp/zkevm.c12a.verifier.dat ${DST}/zkevm.c12a.verifier.dat
cp ${SRC}/zkevm.c12b.verifier_cpp/zkevm.c12b.verifier.dat ${DST}/zkevm.c12b.verifier.dat

#cp ${SRC}/zkevm.chelpers/zkevm.chelpers.* ${DST}/../src/starkpil/src/
#cp ${SRC}/zkevm.c12a.chelpers/zkevm.c12a.chelpers.* ${DST}/../src/starkpil/src/
#cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.* ${DST}/../src/starkpil/src/

cp ${SRC}/zkevm.g16.0001.zkey ${DST}/zkevm.g16.0001.zkey
cp ${SRC}/zkevm.g16.verkey.json ${DST}/zkevm.g16.verkey.json

cp ${SRC}/rom.json ${DST}/rom.json
#cp ${SRC}/zkevm.c12a.verifier_cpp/zkevm.c12a.verifier.cpp ${DST}/../src/starkpil/test/zkevm_c12a_verifier_cpp/zkevm.c12a.verifier.cpp
#cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step2.cpp ${DST}/../src/starkpil/test/zkevm.c12b.chelpers.step2.cpp
#cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step3.cpp ${DST}/../src/starkpil/test/zkevm.c12b.chelpers.step3.cpp
#cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step4.cpp ${DST}/../src/starkpil/test/zkevm.c12b.chelpers.step4.cpp
#cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step42ns.cpp ${DST}/../src/starkpil/test/zkevm.c12b.chelpers.step42ns.cpp
#cp ${SRC}/zkevm.c12b.chelpers/zkevm.c12b.chelpers.step52ns.cpp ${DST}/../src/starkpil/test/zkevm.c12b.chelpers.step52ns.cpp
