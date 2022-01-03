#ifndef PUBLIC_INPUTS_HPP
#define PUBLIC_INPUTS_HPP

#include <string>

using namespace std;

class PublicInputs
{
public:
    string oldStateRoot;
    string oldLocalExitRoot;
    string newStateRoot;
    string newLocalExitRoot;
    string sequencerAddr;
    string batchHashData;
    uint32_t chainId;
    uint32_t batchNum;
};

#endif