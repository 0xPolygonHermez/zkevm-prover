#ifndef DATA_STREAM_HPP
#define DATA_STREAM_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "zkresult.hpp"

using namespace std;

class DataStreamTx
{
public:
    uint8_t gasPricePercentage;
    bool isValid; // Intrinsic
    string stateRoot; // 32 bytes = 64 characters
    string encodedTx; // byte array
    DataStreamTx() : gasPricePercentage(0), isValid(false) {};
    string toString (void)
    {
        return
            "gasPricePercentage=" + to_string(gasPricePercentage) +
            " isValid=" + to_string(isValid) +
            " stateRoot=" + stateRoot +
            " encodedTx.size=" + to_string(encodedTx.size());
    }
};

class DataStreamBlock
{
public:
    uint64_t blockNumber;
    uint64_t timestamp;
    string l1BlockHash; // 32 bytes = 64 characters
    string globalExitRoot; // 32 bytes = 64 characters
    string coinbase; // 20 bytes = 40 characters
    uint64_t forkId;
    string l2BlockHash; // 32 bytes = 64 characters
    string stateRoot; // 32 bytes = 64 characters
    vector<DataStreamTx> txs;
    DataStreamBlock() : blockNumber(0), timestamp(0), forkId(0) {};
    string toString(void)
    {
        return
            "blockNumber=" + to_string(blockNumber) +
            " timestamp=" + to_string(timestamp) +
            " l1BlockHash=" + l1BlockHash +
            " globalExitRoot=" + globalExitRoot +
            " coinbase=" + coinbase +
            " forkId=" + to_string(forkId) +
            " l2BlockHash=" + l2BlockHash +
            " stateRoot=" + stateRoot +
            " txs.size=" + to_string(txs.size());
    }
};

class DataStreamBatch
{
public:
    uint64_t batchNumber;
    vector<DataStreamBlock> blocks; // In order of appearance, block numbers must be consecutive: N, N+1, N+2, etc.
    uint64_t forkId; // It comes from the blocks, and must be the same in all blocks
    DataStreamBatch() { reset(); };
    void reset (void)
    {
        batchNumber = 0;
        blocks.clear();
        forkId = 0;
    }
    string toString (void)
    {
        return
            "batchNumber=" + to_string(batchNumber) +
            " blocks.size=" + to_string(blocks.size());
    }
};

// Decodes a data stream and stores content in a DataStreamBatch
zkresult dataStream2batch (const string &dataStream, DataStreamBatch &batch);

// Encodes a DataStreamBatch into a batch L2 data byte array
zkresult dataStreamBatch2batchL2Data (const DataStreamBatch &batch, string &batchL2Data);

#endif