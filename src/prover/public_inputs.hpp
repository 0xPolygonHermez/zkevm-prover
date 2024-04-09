#ifndef PUBLIC_INPUTS_HPP
#define PUBLIC_INPUTS_HPP

#include <string>
#include <gmp.h>

using namespace std;

class ForcedData
{
public:
    mpz_class globalExitRoot;
    mpz_class blockHashL1;
    uint64_t minTimestamp;
    ForcedData() : minTimestamp(0) {};
};

class PublicInputs
{
public:
    uint64_t   forkID;
    mpz_class  oldStateRoot;
    mpz_class  oldAccInputHash;
    uint32_t   oldBatchNum;
    uint64_t   chainID;
    string     batchL2Data; // This is, in fact, a byte array, not a hex string(not "0xf355...")
    ForcedData forcedData;
    mpz_class  globalExitRoot; // Used when forkID <= 6
    uint64_t   timestamp; // Used when forkID <= 6
    mpz_class  sequencerAddr;
    mpz_class  aggregatorAddress; // Ethereum address of the aggregator that sends verifyBatch TX to the SC, used to prevent proof front-running
    string     witness; // Byte array of the SMT required data in witness (binary) format
    string     dataStream; // Byte array of the batch input required data in Data Streadm (binary) format

    // Etrog batch data (forkID >= 7)
    mpz_class  l1InfoRoot;
    uint64_t   timestampLimit;
    mpz_class  forcedBlockHashL1;

    // Feijoa batch data (forkID >= 10, V3)
    mpz_class  previousL1InfoTreeRoot;
    uint32_t   previousL1InfoTreeIndex;
    mpz_class  forcedHashData;

    // Feijoa blob inner data (forkID >= 10, V3)
    uint32_t   blobType;
    mpz_class  oldBlobStateRoot;
    mpz_class  oldBlobAccInputHash;
    uint64_t   oldBlobNum;
    uint32_t   lastL1InfoTreeIndex;
    mpz_class  lastL1InfoTreeRoot;
    uint64_t   zkGasLimit;
    mpz_class  pointZ;
    mpz_class  pointY;
    string     blobData;
    mpz_class  blobL2HashData;
    mpz_class  batchHashData;

    PublicInputs() :
        forkID(0),
        oldBatchNum(0),
        chainID(0),
        timestamp(0),
        // Etrog batch data:
        timestampLimit(0),
        // Feijoa batch data:
        previousL1InfoTreeIndex(0),
        // Feijoa blob inner data:
        blobType(0),
        oldBlobNum(0),
        lastL1InfoTreeIndex(0),
        zkGasLimit(0)
    {
        aggregatorAddress.set_str("f39fd6e51aad88f6f4ce6ab8827279cfffb92266", 16); // Default aggregator address
    }
};

#endif