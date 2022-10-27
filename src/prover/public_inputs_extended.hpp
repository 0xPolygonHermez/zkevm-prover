#ifndef PUBLIC_INPUTS_EXTENDED
#define PUBLIC_INPUTS_EXTENDED

#include <string>
#include <gmpxx.h>
#include "public_inputs.hpp"

using namespace std;

class PublicInputsExtended
{
public:
    PublicInputs publicInputs;
    string inputHash;
    string newAccInputHash;
    uint32_t newBatchNum;
    string newLocalExitRoot;
    string newStateRoot;
    PublicInputsExtended() : newBatchNum(0) {};
};

#endif