#!/bin/bash -x

VERSION=v3.0.0-RC3-fork.6
FORK_VERSION=$(sed -e 's/.*-fork.//g' <<< ${VERSION})
FORK_ID=fork_$FORK_VERSION

WORKING_DIR=/releases/${VERSION}
CONFIG_DIR=${WORKING_DIR}/config/
C_FILES=${WORKING_DIR}/c_files
CIRCOM_HEADER="#pragma GCC diagnostic push\n#pragma GCC diagnostic ignored \"-Wunused-variable\"\n#pragma GCC push_options\n#pragma GCC optimize (\"O0\")\n#include <stdio.h>\n#include <iostream>\n#include <assert.h>\n#include <cassert>\n"

ZKEVM_VERIFIER_CPP=./src/starkpil/zkevm/witness/zkevm.verifier.cpp
RECURSIVE1_CPP=./src/starkpil/starkRecursive1/witness/recursive1.verifier.cpp
RECURSIVE2_CPP=./src/starkpil/starkRecursive2/witness/recursive2.verifier.cpp
RECURSIVEF_CPP=./src/starkpil/starkRecursiveF/witness/recursivef.verifier.cpp
RECURSIVEFINAL_CPP=./src/starkpil/recursivefinal/final.verifier.cpp

#Sync the config directory
rsync -avz --progress ${CONFIG_DIR}/scripts/ config/scripts/
rsync -avz --progress ${CONFIG_DIR}/ config/

#Uncomment the following line if you want to generate source code the first time after the release files generation

#Copy the chelpers files
cp ${C_FILES}/zkevm.chelpers/*.step* ./src/starkpil/zkevm/chelpers/
cp ${C_FILES}/c12a.chelpers/*.step* ./src/starkpil/starkC12a/chelpers/
cp ${C_FILES}/recursive1.chelpers/*.step* ./src/starkpil/starkRecursive1/chelpers/
cp ${C_FILES}/recursive2.chelpers/*.step* ./src/starkpil/starkRecursive2/chelpers/
cp ${C_FILES}/recursivef.chelpers/*.step* ./src/starkpil/starkRecursiveF/chelpers/

# Generate the zkevm.verifier.cpp
cp ${C_FILES}/zkevm.verifier_cpp/zkevm.verifier.cpp ${ZKEVM_VERIFIER_CPP}
sed -i '1d;2d;3d;4d;5d' ${ZKEVM_VERIFIER_CPP}
sed -i 's/Fr/FrG/g' ${ZKEVM_VERIFIER_CPP}
sed -i "1s/^/#include \"circom.hpp\"\n#include \"calcwit.hpp\"\nnamespace Circom\n{\n/" ${ZKEVM_VERIFIER_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${ZKEVM_VERIFIER_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${ZKEVM_VERIFIER_CPP}

# Generate the recursive1.verifier.cpp
cp ${C_FILES}/recursive1_cpp/recursive1.cpp ${RECURSIVE1_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVE1_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVE1_CPP}
sed -i "1s/^/#include \"circom.recursive1.hpp\"\n#include \"calcwit.recursive1.hpp\"\nnamespace CircomRecursive1\n{\n/" ${RECURSIVE1_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVE1_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVE1_CPP}

# Generate the recursive2.verifier.cpp
cp ${C_FILES}/recursive2_cpp/recursive2.cpp ${RECURSIVE2_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVE2_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVE2_CPP}
sed -i "1s/^/#include \"circom.recursive2.hpp\"\n#include \"calcwit.recursive2.hpp\"\nnamespace CircomRecursive2\n{\n/" ${RECURSIVE2_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVE2_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVE2_CPP}

# Generate the recursivef.verifier.cpp
cp ${C_FILES}/recursivef_cpp/recursivef.cpp ${RECURSIVEF_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVEF_CPP}
sed -i 's/Fr/FrG/g' ${RECURSIVEF_CPP}
sed -i "1s/^/#include \"circom.recursiveF.hpp\"\n#include \"calcwit.recursiveF.hpp\"\nnamespace CircomRecursiveF\n{\n/" ${RECURSIVEF_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVEF_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVEF_CPP}

# Generate the final.verifier.cpp
cp ${C_FILES}/final_cpp/final.cpp ${RECURSIVEFINAL_CPP}
sed -i '1d;2d;3d;4d;5d' ${RECURSIVEFINAL_CPP}
sed -i "1s/^/#include \"circom.final.hpp\"\n#include \"calcwit.final.hpp\"\nnamespace CircomFinal\n{\n/" ${RECURSIVEFINAL_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${RECURSIVEFINAL_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${RECURSIVEFINAL_CPP}

#Copy pols_generated files
cp -r ${CONFIG_DIR}/scripts/* ./src/main_sm/$FORK_ID/scripts/
cp ${WORKING_DIR}/pil/zkevm/main.pil.json  ./src/main_sm/$FORK_ID/scripts/

#main generator files
make main_generator

./build/mainGenerator

make pols_generator

./build/polsGenerator