#ifndef PUBLIC_INPUTS_HPP
#define PUBLIC_INPUTS_HPP

#include <string>
#include <gmp.h>

using namespace std;

class PublicInputs
{
public:
    uint64_t  forkID;
    mpz_class oldStateRoot;
    mpz_class oldAccInputHash;
    uint32_t  oldBatchNum;
    uint64_t  chainID;
    string    batchL2Data; // This is, in fact, a byte array, not a hex string(not "0xf355...")
    mpz_class globalExitRoot; // Used when forkID <= 6
    mpz_class l1InfoRoot; // Used when forkID >= 7
    uint64_t  timestamp; // Used when forkID <= 6
    uint64_t  timestampLimit; // Used when forkID >= 7
    mpz_class forcedBlockHashL1; // Used when forkID >= 7
    mpz_class sequencerAddr;
    mpz_class aggregatorAddress; // Ethereum address of the aggregator that sends verifyBatch TX to the SC, used to prevent proof front-running
    string    witness; // Byte array of the SMT required data in witness (binary) format
    string    dataStream; // Byte array of the batch input required data in Data Streadm (binary) format

    PublicInputs() : forkID(0), oldBatchNum(0), chainID(0), timestamp(0), timestampLimit(0)
    {
        aggregatorAddress.set_str("f39fd6e51aad88f6f4ce6ab8827279cfffb92266", 16); // Default aggregator address
    }

    bool operator==(PublicInputs &publicInputs)
    {
        return
            oldStateRoot      == publicInputs.oldStateRoot &&
            oldAccInputHash   == publicInputs.oldAccInputHash &&
            oldBatchNum       == publicInputs.oldBatchNum &&
            chainID           == publicInputs.chainID &&
            forkID            == publicInputs.forkID &&
            batchL2Data       == publicInputs.batchL2Data &&
            globalExitRoot    == publicInputs.globalExitRoot &&
            timestamp         == publicInputs.timestamp &&
            sequencerAddr     == publicInputs.sequencerAddr &&
            aggregatorAddress == publicInputs.aggregatorAddress;
    }
};

#endif