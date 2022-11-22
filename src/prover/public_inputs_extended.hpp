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

    string    inputHash;
    mpz_class newAccInputHash;
    uint32_t  newBatchNum;
    mpz_class newLocalExitRoot;
    mpz_class newStateRoot;
    
    PublicInputsExtended() : newBatchNum(0) {};

    bool operator==(PublicInputsExtended &publicInputsExtended)
    {
        return
            publicInputs     == publicInputsExtended.publicInputs &&
            inputHash        == publicInputsExtended.inputHash &&
            newAccInputHash  == publicInputsExtended.newAccInputHash &&
            newBatchNum      == publicInputsExtended.newBatchNum &&
            newLocalExitRoot == publicInputsExtended.newLocalExitRoot &&
            newStateRoot     == publicInputsExtended.newStateRoot;
    }
};

#endif