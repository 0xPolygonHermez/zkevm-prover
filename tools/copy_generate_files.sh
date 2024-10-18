#!/bin/bash -x

VERSION=v9.0.0-rc.1-fork.13
FORK_VERSION=$(sed -e 's/.*-fork.//g' <<< ${VERSION})
FORK_ID=fork_$FORK_VERSION
PARENT_FORK_ID=fork_13
EXCLUDE_CONSTTREE="true"

WORKING_DIR=/releases/${VERSION}
CONFIG_DIR=${WORKING_DIR}/config/
C_FILES=${WORKING_DIR}/c_files
CIRCOM_HEADER="#pragma GCC diagnostic push\n#pragma GCC diagnostic ignored \"-Wunused-variable\"\n#pragma GCC push_options\n#pragma GCC optimize (\"O0\")\n#include <stdio.h>\n#include <iostream>\n#include <assert.h>\n#include <cassert>\n"

ZKEVM_VERIFIER_CPP=./src/starkpil/zkevm/witness/fork_${FORK_VERSION}/zkevm.verifier.cpp
RECURSIVE1_CPP=./src/starkpil/starkRecursive1/witness/fork_${FORK_VERSION}/recursive1.verifier.cpp
RECURSIVE2_CPP=./src/starkpil/starkRecursive2/witness/fork_${FORK_VERSION}/recursive2.verifier.cpp
RECURSIVEF_CPP=./src/starkpil/starkRecursiveF/witness/fork_${FORK_VERSION}/recursivef.verifier.cpp
RECURSIVEFINAL_CPP=./src/starkpil/recursivefinal/witness/fork_${FORK_VERSION}/final.verifier.cpp

EXCLUDE_OPTION=""

# Check if we need to exclude .consttree files
if [ "$EXCLUDE_CONSTTREE" = true ]; then
    EXCLUDE_OPTION="--exclude '*.consttree'"
fi

#Sync the config directory
rsync -avz --progress ${CONFIG_DIR}/scripts/ config/scripts/
rsync -avz --progress ${EXCLUDE_OPTION} ${CONFIG_DIR}/ config/
rm config/scripts/rom.json
rm config/scripts/metadata-rom.txt

# Copy setup files
cp ${CONFIG_DIR}/../build/sha256.txt src/config/setup-${FORK_VERSION}.txt

#Uncomment the following line if you want to generate source code the first time after the release files generation

#Copy the chelpers files
cp ${C_FILES}/zkevm.chelpers/ZkevmSteps.hpp ./src/starkpil/zkevm/chelpers/ZkevmSteps.hpp
cp ${C_FILES}/c12a.chelpers/C12aSteps.hpp ./src/starkpil/starkC12a/chelpers/C12aSteps.hpp
cp ${C_FILES}/recursive1.chelpers/Recursive1Steps.hpp ./src/starkpil/starkRecursive1/chelpers/Recursive1Steps.hpp
cp ${C_FILES}/recursive2.chelpers/Recursive2Steps.hpp ./src/starkpil/starkRecursive2/chelpers/Recursive2Steps.hpp
cp ${C_FILES}/recursivef.chelpers/RecursiveFSteps.hpp ./src/starkpil/starkRecursiveF/chelpers/RecursiveFSteps.hpp

# Generate the zkevm.verifier.cpp
cp ${C_FILES}/zkevm.verifier_cpp/zkevm.verifier.cpp ${ZKEVM_VERIFIER_CPP}
sed -i '1d;2d;3d;4d;5d' ${ZKEVM_VERIFIER_CPP}
sed -i 's/Fr/FrG/g' ${ZKEVM_VERIFIER_CPP}
sed -i "1s/^/#include \"circom.hpp\"\n#include \"calcwit.hpp\"\nnamespace CircomFork${FORK_VERSION}\n{\n/" ${ZKEVM_VERIFIER_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${ZKEVM_VERIFIER_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${ZKEVM_VERIFIER_CPP}

# Generate the recursive1.verifier.cpp
cp ${C_FILES}/recursive1_cpp/recursive1.cpp ${RECURSIVE1_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVE1_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVE1_CPP}
sed -i "1s/^/#include \"circom.recursive1.hpp\"\n#include \"calcwit.recursive1.hpp\"\nnamespace CircomRecursive1Fork${FORK_VERSION}\n{\n/" ${RECURSIVE1_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVE1_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVE1_CPP}

# Generate the recursive2.verifier.cpp
cp ${C_FILES}/recursive2_cpp/recursive2.cpp ${RECURSIVE2_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVE2_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVE2_CPP}
sed -i "1s/^/#include \"circom.recursive2.hpp\"\n#include \"calcwit.recursive2.hpp\"\nnamespace CircomRecursive2Fork${FORK_VERSION}\n{\n/" ${RECURSIVE2_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVE2_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVE2_CPP}

# Generate the recursivef.verifier.cpp
cp ${C_FILES}/recursivef_cpp/recursivef.cpp ${RECURSIVEF_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVEF_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVEF_CPP}
sed -i "1s/^/#include \"circom.recursiveF.hpp\"\n#include \"calcwit.recursiveF.hpp\"\nnamespace CircomRecursiveFFork${FORK_VERSION}\n{\n/" ${RECURSIVEF_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVEF_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVEF_CPP}

# Generate the final.verifier.cpp
cp ${C_FILES}/final_cpp/final.cpp ${RECURSIVEFINAL_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVEFINAL_CPP}
sed -i "1s/^/#include \"circom.final.hpp\"\n#include \"calcwit.final.hpp\"\nnamespace CircomFinalFork${FORK_VERSION}\n{\n/" ${RECURSIVEFINAL_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVEFINAL_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVEFINAL_CPP}

#Copy pols_generated files
cp ${CONFIG_DIR}/scripts/rom.json ./src/main_sm/$PARENT_FORK_ID/scripts/rom_${FORK_VERSION}.json
cp ${CONFIG_DIR}/scripts/metadata-rom.txt ./src/main_sm/$PARENT_FORK_ID/scripts/metadata-rom_${FORK_VERSION}.txt
if [ "$FORK_ID" == "$PARENT_FORK_ID" ]; then
    cp "${WORKING_DIR}/pil/zkevm/main.pil.json" "./src/main_sm/$PARENT_FORK_ID/scripts/"
fi

#main generator files
make generate

#pols generator files
make pols_generator
./build/polsGenerator