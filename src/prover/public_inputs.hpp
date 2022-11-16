#ifndef PUBLIC_INPUTS_HPP
#define PUBLIC_INPUTS_HPP

#include <string>

using namespace std;

class PublicInputs
{
public:
    string oldStateRoot;
    string oldAccInputHash;
    uint32_t oldBatchNum;
    uint64_t chainID;
    string batchL2Data;
    string globalExitRoot;
    uint64_t timestamp;
    string sequencerAddr;
    string aggregatorAddress; // Ethereum address of the aggregator that sends verifyBatch TX to the SC, used to prevent proof front-running

    PublicInputs() : oldBatchNum(0), chainID(0), timestamp(0) {;}

    bool operator==(PublicInputs &publicInputs)
    {
        return
            oldStateRoot == publicInputs.oldStateRoot &&
            oldAccInputHash == publicInputs.oldAccInputHash &&
            oldBatchNum == publicInputs.oldBatchNum &&
            chainID == publicInputs.chainID &&
            batchL2Data == publicInputs.batchL2Data &&
            globalExitRoot == publicInputs.globalExitRoot &&
            timestamp == publicInputs.timestamp &&
            sequencerAddr == publicInputs.sequencerAddr &&
            aggregatorAddress == publicInputs.aggregatorAddress;
    }
};

#endif