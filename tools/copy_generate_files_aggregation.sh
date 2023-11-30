#!/bin/bash -x

VERSION=v3.0.0-RC3-fork.6
FORK_VERSION=$(sed -e 's/.*-fork.//g' <<< ${VERSION})
FORK_ID=fork_$FORK_VERSION

WORKING_DIR=/home/edu/roger/aggregation
CONFIG_DIR=${WORKING_DIR}/config/
C_FILES=${WORKING_DIR}/c_files
CIRCOM_HEADER="#pragma GCC diagnostic push\n#pragma GCC diagnostic ignored \"-Wunused-variable\"\n#pragma GCC push_options\n#pragma GCC optimize (\"O0\")\n#include <stdio.h>\n#include <iostream>\n#include <assert.h>\n#include <cassert>\n"

MULTICHAIN_PREP_CPP=./src/aggregation/starkMultichainPrep/witness/multichainPrep.verifier.cpp
MULTICHAIN_AGG_CPP=./src/aggregation/starkMultichainAgg/witness/multichainAgg.verifier.cpp
MULTICHAIN_AGG_F_CPP=./src/aggregation/starkMultichainAggF/witness/multichainAggF.verifier.cpp
MULTICHAIN_FINAL_CPP=./src/aggregation/multichainFinal/multichainFinal.verifier.cpp

#Sync the config directory
rsync -avz --progress ${CONFIG_DIR}/scripts/ config/scripts/
rsync -avz --progress ${CONFIG_DIR}/ config/

#Uncomment the following line if you want to generate source code the first time after the release files generation

#Copy the chelpers files
cp ${C_FILES}/multichainPrep.chelpers/*.step* ./src/aggregation/starkMultichainPrep/chelpers/
cp ${C_FILES}/multichainAgg.chelpers/*.step* ./src/aggregation/starkMultichainAgg/chelpers/
cp ${C_FILES}/multichainAggF.chelpers/*.step* ./src/aggregation/starkMultichainAggF/chelpers/

# Generate the multichainPrep.verifier.cpp
cp ${C_FILES}/multichainPrep_cpp/multichainPrep.cpp ${MULTICHAIN_PREP_CPP}
sed -i '1d;2d;3d;4d;5d' ${MULTICHAIN_PREP_CPP}
sed -i 's/Fr/FrG/g' ${MULTICHAIN_PREP_CPP}
sed -i "1s/^/#include \"circom.multichainPrep.hpp\"\n#include \"calcwit.multichainPrep.hpp\"\nnamespace CircomMultichainPrep\n{\n/" ${MULTICHAIN_PREP_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${MULTICHAIN_PREP_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${MULTICHAIN_PREP_CPP}

# Generate the multichainAgg.verifier.cpp
cp ${C_FILES}/multichainAgg_cpp/multichainAgg.cpp ${MULTICHAIN_AGG_CPP}
sed -i '1d;2d;3d;4d;5d' ${MULTICHAIN_AGG_CPP}
sed -i 's/Fr/FrG/g' ${MULTICHAIN_AGG_CPP}
sed -i "1s/^/#include \"circom.multichainAgg.hpp\"\n#include \"calcwit.multichainAgg.hpp\"\nnamespace CircomMultichainAgg\n{\n/" ${MULTICHAIN_AGG_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${MULTICHAIN_AGG_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${MULTICHAIN_AGG_CPP}

# Generate the multichainAggF.verifier.cpp
cp ${C_FILES}/multichainAggF_cpp/multichainAggF.cpp ${MULTICHAIN_AGG_F_CPP}
sed -i '1d;2d;3d;4d;5d' ${MULTICHAIN_AGG_F_CPP}
sed -i 's/Fr/FrG/g' ${MULTICHAIN_AGG_F_CPP}
sed -i "1s/^/#include \"circom.multichainAggF.hpp\"\n#include \"calcwit.multichainAggF.hpp\"\nnamespace CircomMultichainAggF\n{\n/" ${MULTICHAIN_AGG_F_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${MULTICHAIN_AGG_F_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${MULTICHAIN_AGG_F_CPP}

# Generate the final.verifier.cpp
cp ${C_FILES}/multichainFinal_cpp/multichainFinal.cpp ${MULTICHAIN_FINAL_CPP}
sed -i '1d;2d;3d;4d;5d' ${MULTICHAIN_FINAL_CPP}
sed -i "1s/^/#include \"circom.multichainFinal.hpp\"\n#include \"calcwit.multichainFinal.hpp\"\nnamespace CircomMultichainFinal\n{\n/" ${MULTICHAIN_FINAL_CPP}
sed -i "1s/^/$CIRCOM_HEADER/" ${MULTICHAIN_FINAL_CPP} 
echo -e "}\n#pragma GCC diagnostic pop" >> ${MULTICHAIN_FINAL_CPP}