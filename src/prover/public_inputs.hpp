#ifndef PUBLIC_INPUTS_HPP
#define PUBLIC_INPUTS_HPP

#include <string>

using namespace std;

class PublicInputs
{
public:
    string oldStateRoot;
    string newStateRoot;
    string oldLocalExitRoot;
    string newLocalExitRoot;
    string sequencerAddr;
    string batchHashData;
    uint32_t defaultChainId;
    uint32_t batchNum;
    uint32_t blockNum;
    uint64_t timestamp;

    PublicInputs() : defaultChainId(0), batchNum(0), blockNum(0), timestamp(0) {;}
};

#endif