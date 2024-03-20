#!/bin/bash -x

VERSION=develop_feijoa-fork.9
FORK_VERSION=$(sed -e 's/.*-fork.//g' <<< ${VERSION})
FORK_ID=fork_$FORK_VERSION

WORKING_DIR=/releases/${VERSION}
CONFIG_DIR=${WORKING_DIR}/config/
C_FILES=${WORKING_DIR}/c_files
CIRCOM_HEADER="#pragma GCC diagnostic push\n#pragma GCC diagnostic ignored \"-Wunused-variable\"\n#pragma GCC push_options\n#pragma GCC optimize (\"O0\")\n#include <stdio.h>\n#include <iostream>\n#include <assert.h>\n#include <cassert>\n"

ZKEVM_VERIFIER_CPP=./src/starkpil/zkevm/witness/zkevm.verifier.cpp
RECURSIVE1_CPP=./src/starkpil/starkRecursive1/witness/recursive1.cpp
RECURSIVE2_CPP=./src/starkpil/starkRecursive2/witness/recursive2.cpp

BLOB_INNER_VERIFIER_CPP=./src/starkpil/blob_inner/witness/blob_inner.verifier.cpp
BLOB_INNER_RECURSIVE1_CPP=./src/starkpil/blob_inner_recursive1/witness/blob_inner_recursive1.cpp
BLOB_OUTER_CPP=./src/starkpil/blob_outer/witness/blob_outer.cpp
BLOB_OUTER_RECURSIVE2_CPP=./src/starkpil/blob_outer_recursive2/witness/blob_outer_recursive2.cpp

RECURSIVEF_CPP=./src/starkpil/starkRecursiveF/witness/recursivef.cpp
RECURSIVEFINAL_CPP=./src/starkpil/recursivefinal/final.cpp

#Sync the config directory
rsync -avz --progress ${CONFIG_DIR}/scripts/ config/scripts/
rsync -avz --progress ${CONFIG_DIR}/ config/
rm config/scripts/rom.json
rm config/scripts/metadata-rom.txt

rm config/scripts/blob-rom.json
rm config/scripts/metadata-blob-rom.txt

#Uncomment the following line if you want to generate source code the first time after the release files generation

#Copy the chelpers files
cp ${C_FILES}/zkevm.chelpers/ZkevmSteps.hpp ./src/starkpil/zkevm/chelpers/ZkevmSteps.hpp
cp ${C_FILES}/c12a.chelpers/C12aSteps.hpp ./src/starkpil/starkC12a/chelpers/C12aSteps.hpp
cp ${C_FILES}/recursive1.chelpers/Recursive1Steps.hpp ./src/starkpil/starkRecursive1/chelpers/Recursive1Steps.hpp
cp ${C_FILES}/recursive2.chelpers/Recursive2Steps.hpp ./src/starkpil/starkRecursive2/chelpers/Recursive2Steps.hpp

cp ${C_FILES}/blob_inner.chelpers/BlobInnerSteps.hpp ./src/starkpil/blob_inner/chelpers/BlobInnerSteps.hpp
cp ${C_FILES}/blob_inner_compressor.chelpers/BlobInnerCompressorSteps.hpp ./src/starkpil/blob_inner_compressor/chelpers/BlobInnerCompressorSteps.hpp
cp ${C_FILES}/blob_inner_recursive1.chelpers/BlobInnerRecursive1Steps.hpp ./src/starkpil/blob_inner_recursive1/chelpers/BlobInnerRecursive1Steps.hpp
cp ${C_FILES}/blob_outer.chelpers/BlobOuterSteps.hpp ./src/starkpil/blob_outer/chelpers/BlobOuterSteps.hpp
cp ${C_FILES}/blob_outer_recursive2.chelpers/BlobOuterRecursive2Steps.hpp ./src/starkpil/blob_outer_recursive2/chelpers/BlobOuterRecursive2Steps.hpp

cp ${C_FILES}/recursivef.chelpers/RecursiveFSteps.hpp ./src/starkpil/starkRecursiveF/chelpers/RecursiveFSteps.hpp

# Generate the zkevm.verifier.cpp
cp ${C_FILES}/zkevm.verifier_cpp/zkevm.verifier.cpp ${ZKEVM_VERIFIER_CPP}
sed -i '1d;2d;3d;4d;5d' ${ZKEVM_VERIFIER_CPP}
sed -i 's/Fr/FrG/g' ${ZKEVM_VERIFIER_CPP}
sed -i "1s/^/#include \"circom.hpp\"\n#include \"calcwit.hpp\"\nnamespace Circom\n{\n/" ${ZKEVM_VERIFIER_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${ZKEVM_VERIFIER_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${ZKEVM_VERIFIER_CPP}

# Generate the recursive1.cpp
cp ${C_FILES}/recursive1_cpp/recursive1.cpp ${RECURSIVE1_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVE1_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVE1_CPP}
sed -i "1s/^/#include \"circom.recursive1.hpp\"\n#include \"calcwit.recursive1.hpp\"\nnamespace CircomRecursive1\n{\n/" ${RECURSIVE1_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVE1_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVE1_CPP}

# Generate the recursive2.cpp
cp ${C_FILES}/recursive2_cpp/recursive2.cpp ${RECURSIVE2_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVE2_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVE2_CPP}
sed -i "1s/^/#include \"circom.recursive2.hpp\"\n#include \"calcwit.recursive2.hpp\"\nnamespace CircomRecursive2\n{\n/" ${RECURSIVE2_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVE2_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVE2_CPP}

# Generate the blob_inner.verifier.cpp
cp ${C_FILES}/blob_inner.verifier_cpp/blob_inner.verifier.cpp ${BLOB_INNER_VERIFIER_CPP}
sed -i '1d;2d;3d;4d;5d' ${BLOB_INNER_VERIFIER_CPP}
sed -i 's/Fr/FrG/g' ${BLOB_INNER_VERIFIER_CPP}
sed -i "1s/^/#include \"circom.blob_inner.hpp\"\n#include \"calcwit.blob_inner.hpp\"\nnamespace CircomBlobInner\n{\n/" ${BLOB_INNER_VERIFIER_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${BLOB_INNER_VERIFIER_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${BLOB_INNER_VERIFIER_CPP}

# Generate the blob_inner_recursive1.cpp
cp ${C_FILES}/recursive1_cpp/recursive1.cpp ${BLOB_INNER_RECURSIVE1_CPP}
sed -i '1d;2d;3d;4d;5d' ${BLOB_INNER_RECURSIVE1_CPP}
sed -i 's/Fr/FrG/g' ${BLOB_INNER_RECURSIVE1_CPP}
sed -i "1s/^/#include \"circom.blob_inner_recursive1.hpp\"\n#include \"calcwit.blob_inner_recursive1.hpp\"\nnamespace CircomBlobInnerRecursive1\n{\n/" ${BLOB_INNER_RECURSIVE1_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${BLOB_INNER_RECURSIVE1_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${BLOB_INNER_RECURSIVE1_CPP}

# Generate the blob_outer.cpp
cp ${C_FILES}/blob_outer_cpp/blob_outer.cpp ${BLOB_OUTER_CPP}
sed -i '1d;2d;3d;4d;5d' ${BLOB_OUTER_CPP}
sed -i 's/Fr/FrG/g' ${BLOB_OUTER_CPP}
sed -i "1s/^/#include \"circom.blob_outer.hpp\"\n#include \"calcwit.blob_outer.hpp\"\nnamespace CircomBlobOuter\n{\n/" ${BLOB_OUTER_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${BLOB_OUTER_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${BLOB_OUTER_CPP}

# Generate the blob_outer_recursive2.cpp
cp ${C_FILES}/blob_outer_recursive2_cpp/blob_outer_recursive2.cpp ${BLOB_OUTER_RECURSIVE2_CPP}
sed -i '1d;2d;3d;4d;5d' ${BLOB_OUTER_RECURSIVE2_CPP}
sed -i 's/Fr/FrG/g' ${BLOB_OUTER_RECURSIVE2_CPP}
sed -i "1s/^/#include \"circom.blob_outer_recursive2.hpp\"\n#include \"calcwit.blob_outer_recursive2.hpp\"\nnamespace CircomBlobOuterRecursive2\n{\n/" ${BLOB_OUTER_RECURSIVE2_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${BLOB_OUTER_RECURSIVE2_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${BLOB_OUTER_RECURSIVE2_CPP}

# Generate the recursivef.cpp
cp ${C_FILES}/recursivef_cpp/recursivef.cpp ${RECURSIVEF_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVEF_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVEF_CPP}
sed -i "1s/^/#include \"circom.recursiveF.hpp\"\n#include \"calcwit.recursiveF.hpp\"\nnamespace CircomRecursiveF\n{\n/" ${RECURSIVEF_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVEF_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVEF_CPP}

# Generate the final.cpp
cp ${C_FILES}/final_cpp/final.cpp ${RECURSIVEFINAL_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVEFINAL_CPP}
sed -i "1s/^/#include \"circom.final.hpp\"\n#include \"calcwit.final.hpp\"\nnamespace CircomFinal\n{\n/" ${RECURSIVEFINAL_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVEFINAL_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVEFINAL_CPP}

#Copy pols_generated files
cp ${CONFIG_DIR}/scripts/rom.json ./src/main_sm/$FORK_ID/scripts/
cp ${CONFIG_DIR}/scripts/metadata-rom.txt ./src/main_sm/$FORK_ID/scripts/
cp ${WORKING_DIR}/pil/zkevm/main.pil.json  ./src/main_sm/$FORK_ID/scripts/

cp ${CONFIG_DIR}/scripts/blob-rom.json ./src/main_sm/${FORK_ID}_blob/scripts/rom.json
cp ${CONFIG_DIR}/scripts/metadata-blob-rom.txt ./src/main_sm/${FORK_ID}_blob/scripts/metadata-rom.txt
cp ${WORKING_DIR}/pil/blob_inner/main_blob.pil.json  ./src/main_sm/${FORK_ID}_blob/scripts/main.pil.json

#main generator files
make generate

#pols generator files
make pols_generator
./build/polsGenerator