#ifndef PUBLIC_INPUTS_HPP
#define PUBLIC_INPUTS_HPP

#include <string>
#include <gmp.h>

using namespace std;

class PublicInputs
{
public:
    mpz_class oldStateRoot;
    mpz_class oldAccInputHash;
    uint32_t  oldBatchNum;
    uint64_t  chainID;
    string    batchL2Data; // This is, in fact, a byte array, not a hex string(not "0xf355...")
    mpz_class globalExitRoot;
    uint64_t  timestamp;
    mpz_class sequencerAddr;
    mpz_class aggregatorAddress; // Ethereum address of the aggregator that sends verifyBatch TX to the SC, used to prevent proof front-running

    PublicInputs() : oldBatchNum(0), chainID(0), timestamp(0) {;}

    bool operator==(PublicInputs &publicInputs)
    {
        return
            oldStateRoot      == publicInputs.oldStateRoot &&
            oldAccInputHash   == publicInputs.oldAccInputHash &&
            oldBatchNum       == publicInputs.oldBatchNum &&
            chainID           == publicInputs.chainID &&
            batchL2Data       == publicInputs.batchL2Data &&
            globalExitRoot    == publicInputs.globalExitRoot &&
            timestamp         == publicInputs.timestamp &&
            sequencerAddr     == publicInputs.sequencerAddr &&
            aggregatorAddress == publicInputs.aggregatorAddress;
    }
};

#endif