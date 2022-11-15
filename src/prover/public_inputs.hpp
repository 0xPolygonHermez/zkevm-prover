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

    PublicInputs() : oldBatchNum(0), chainID(0), timestamp(0), aggregatorAddress("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D") {;}
};

#endif