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
};

class DataStreamBatch
{
public:
    uint64_t batchNumber;
    map<uint64_t, DataStreamBlock> blocks;
    uint64_t currentBlock;
    DataStreamBatch() : batchNumber(0), currentBlock(0) {};
    void reset (void)
    {
        batchNumber = 0;
        blocks.clear();
        currentBlock = 0;
    }
};

zkresult dataStream2data (const string &dataStream, DataStreamBatch &batch);

#endif