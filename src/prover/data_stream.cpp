#include "data_stream.hpp"
#include "zklog.hpp"
#include "scalar.hpp"

#define LOG_DATA_STREAM

uint8_t ParseU8 (const string &data, uint64_t &p)
{
    uint8_t result;
    result = data[p];
    p++;
    return result;
}

uint16_t ParseLittleEndianU16 (const string &data, uint64_t &p)
{
    uint16_t result;
    result = (uint16_t(uint8_t(data[p  ]))<<8) +
              uint16_t(uint8_t(data[p+1]));
    p += 2;
    return result;
}

uint16_t ParseBigEndianU16 (const string &data, uint64_t &p)
{
    uint16_t result;
    result = (uint16_t(uint8_t(data[p+1]))<<8) +
              uint16_t(uint8_t(data[p  ]));
    p += 2;
    return result;
}

uint32_t ParseLittleEndianU32 (const string &data, uint64_t &p)
{
    uint32_t result;
    result = (uint32_t(uint8_t(data[p  ]))<<24) +
             (uint32_t(uint8_t(data[p+1]))<<16) +
             (uint32_t(uint8_t(data[p+2]))<<8) +
              uint32_t(uint8_t(data[p+3]));
    p += 4;
    return result;
}

uint32_t ParseBigEndianU32 (const string &data, uint64_t &p)
{
    uint32_t result;
    result = *(uint32_t *)(&data[p]);
    p += 4;
    return result;
}

uint64_t ParseLittleEndianU64 (const string &data, uint64_t &p)
{
    uint64_t result;
    result = (uint64_t(uint8_t(data[p  ]))<<56) +
             (uint64_t(uint8_t(data[p+1]))<<48) +
             (uint64_t(uint8_t(data[p+2]))<<40) +
             (uint64_t(uint8_t(data[p+3]))<<32) +
             (uint64_t(uint8_t(data[p+4]))<<24) +
             (uint64_t(uint8_t(data[p+5]))<<16) +
             (uint64_t(uint8_t(data[p+6]))<<8) +
              uint64_t(uint8_t(data[p+7]));
    p += 8;
    return result;
}

uint64_t ParseBigEndianU64 (const string &data, uint64_t &p)
{
    uint64_t result;
    result = *(uint64_t *)(&data[p]);
    p += 8;
    return result;
}

string EncodeBigEndianU32 (uint32_t value)
{
    string result;
    result.append((const char *)&value, 4);
    return result;
}

zkresult dataStream2batch (const string &dataStream, DataStreamBatch &batch)
{
    // Initialize variables
    uint64_t p = 0;
    batch.reset();

    // While there is data to process
    while (p < dataStream.size())
    {
        /*
        DATA ENTRY format (FileEntry):
        u8 packetType // 2:Data entry, 0:Padding
        u32 Length // Total length of data entry (17 bytes + length(data))
        u32 Type // 0xb0:Bookmark, 1:Event1, 2:Event2,...
        u64 Number // Entry number (sequential starting with 0)
        u8[] data
        */

        // Parse packet type
        uint8_t packetType = ParseU8(dataStream, p);

        // Parse length
        if (p + 3 >= dataStream.size())
        {
            zklog.error("dataStream2batch() parsing length, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        uint32_t length = ParseLittleEndianU32(dataStream, p);

        // Check length range
        if (length < 17)
        {
            zklog.error("dataStream2batch() checking length range, length=" + to_string(length) + "<17");
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        if (p + length - 1 /*packetType*/ - 4 /*length*/ - 1 >= dataStream.size())
        {
            zklog.error("dataStream2batch() checking length, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            //return ZKR_DATA_STREAM_INVALID_DATA;
        }

        // Parse type
        if (p + 3 >= dataStream.size())
        {
            zklog.error("dataStream2batch() parsing type, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        uint32_t type = ParseLittleEndianU32(dataStream, p);

        // Parse number
        if (p + 8 > dataStream.size())
        {
            zklog.error("dataStream2batch() parsing number, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        uint64_t number = ParseLittleEndianU64(dataStream, p);

        // Check that there is enough room for data
        uint64_t dataLength = length - 17;
        if (p + dataLength - 1 >= dataStream.size())
        {
            zklog.error("dataStream2batch() checking data length=" + to_string(dataLength) + ", run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }

        // If packet type is padding, simply skip data
        if (packetType == 0)
        {
            p += dataLength;
            continue;
        }

        // If packet type is not entry, then fail
        if (packetType != 2)
        {
            zklog.error("dataStream2batch() unsupported packet type=" + to_string(p) + " data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }

        // Check type
        switch (type)
        {
            case 0xb0: // Bookmark type, skip
            {
                p += dataLength;
#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() BOOKMARK");
#endif
                continue;
            }

            /*
            Start L2 Block:
                Entry type = 1
                Entry data:
                    u64 batchNum
                    u64 blockL2Num
                    u64 timestamp
                    u8[32] l1BlockHash
                    u8[32] globalExitRoot
                    u8[20] coinbase
                    u16 forkId
            */
            case 1: // Start L2 block
            {
                // Check data length range
                if (dataLength != 110)
                {
                    zklog.error("dataStream2batch() start L2 block invalid dataLength=" + to_string(dataLength) + "!=110 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get batch number
                uint64_t batchNumber;
                batchNumber = ParseBigEndianU64(dataStream, p);

                // Check it is not zero
                if (batchNumber == 0)
                {
                    zklog.error("dataStream2batch() start L2 block, found batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Create a block and fill it with the entry data
                DataStreamBlock block;
                block.blockNumber = ParseBigEndianU64(dataStream, p);
                if (block.blockNumber == 0)
                {
                    zklog.error("dataStream2batch() end L2 block, found blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                block.timestamp = ParseBigEndianU64(dataStream, p);
                ba2string(block.l1BlockHash, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;
                ba2string(block.globalExitRoot, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;
                ba2string(block.coinbase, (const uint8_t *)dataStream.c_str() + p, 20);
                p += 20;
                block.forkId = ParseBigEndianU16(dataStream, p);

                // If batch number has already been assigned
                if (batch.batchNumber != 0)
                {
                    // Check that the batch numbers match
                    if (batch.batchNumber != batchNumber) // If they don't match, we are getting blocks from different batches, so fail
                    {
                        zklog.error("dataStream2batch() start L2 block, batch number mismatch, batchNumber=" + to_string(batchNumber) + " batch.batchNumber=" + to_string(batch.batchNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                        return ZKR_DATA_STREAM_INVALID_DATA;
                    }
                    
                    // If batch number was not zero, check the block did not exist previously
                    map<uint64_t, DataStreamBlock>::iterator it;
                    it = batch.blocks.find(block.blockNumber);
                    if (it != batch.blocks.end())
                    {
                        zklog.error("dataStream2batch() start L2 block, found existing block number=" + to_string(block.blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                        return ZKR_DATA_STREAM_INVALID_DATA;
                    }
                }
                else
                {
                    // Record the first batch number found
                    batch.batchNumber = batchNumber;
                }

                // Fork ID cannot be zero
                if (block.forkId == 0)
                {
                    zklog.error("dataStream2batch() start L2 block, found forkId=0 block number=" + to_string(block.blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // If fork ID has already been assigned
                if ((batch.forkId != 0) && (block.forkId != batch.forkId))
                {
                    zklog.error("dataStream2batch() start L2 block, found block.forkId=" + to_string(block.forkId) + " different from btach.forkId=" + to_string(batch.forkId) + " blockNumber=" + to_string(block.blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Store block in batch
                batch.blocks[block.blockNumber] = block;
                batch.currentBlock = block.blockNumber;
                batch.forkId = block.forkId;

#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() START L2 BLOCK " + block.toString());
#endif

                continue;
            }

            /*
            L2 TX:
                Entry type = 2
                Entry data:
                    u8 gasPricePercentage
                    u8 isValid // Intrinsic
                    u8[32] stateRoot
                    u32 encodedTXLength
                    u8[] encodedTX
            */
            case 2: // L2 TX
            {
                // Check data length range
                if (dataLength < 38)
                {
                    zklog.error("dataStream2batch() L2 TX invalid dataLength=" + to_string(dataLength) + "<38 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check that batch is in the proper state, i.e. with current block still open
                if (batch.batchNumber == 0)
                {
                    zklog.error("dataStream2batch() L2 TX found batch.batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                if (batch.currentBlock == 0)
                {
                    zklog.error("dataStream2batch() L2 TX found batch.currentBlock=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                map<uint64_t, DataStreamBlock>::iterator it;
                it = batch.blocks.find(batch.currentBlock);
                if (it == batch.blocks.end())
                {
                    zklog.error("dataStream2batch() L2 TX, could not find block batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                if (!it->second.l2BlockHash.empty())
                {
                    zklog.error("dataStream2batch() L2 TX, found current block with l2BlockHash not empty batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                if (!it->second.stateRoot.empty())
                {
                    zklog.error("dataStream2batch() L2 TX, found current block with stateRoot not empty batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Parse Tx
                DataStreamTx tx;
                tx.gasPricePercentage = ParseU8(dataStream, p);
                tx.isValid = ParseU8(dataStream, p);
                ba2string(tx.stateRoot, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;
                uint32_t encodedTxLength = ParseBigEndianU32(dataStream, p);
                if (p + encodedTxLength - 1 >= dataStream.size())
                {
                    zklog.error("dataStream2batch() L2 TX, run out of data batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                tx.encodedTx = dataStream.substr(p, encodedTxLength);
                p += encodedTxLength;

                // Add it to the current block
                it->second.txs.emplace_back(tx);

#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() L2 TX " + tx.toString());
#endif

                continue;
            }

            /*
            End L2 Block:
                Entry type = 3
                Entry data:
                    u64 blockL2Num
                    u8[32] l2BlockHash
                    u8[32] stateRoot
            */
            case 3: // End L2 Block
            {
                // Check data length range
                if (dataLength != 72)
                {
                    zklog.error("dataStream2batch() end L2 block invalid dataLength=" + to_string(dataLength) + "!=72 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get block number
                uint64_t blockNumber = ParseBigEndianU64(dataStream, p);

                // Check it is not zero
                if (blockNumber == 0)
                {
                    zklog.error("dataStream2batch() end L2 block, found blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check that batch number has already been assigned
                if (batch.batchNumber == 0)
                {
                    zklog.error("dataStream2batch() end L2 block, found batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                    
                // Check the block did exist previously
                map<uint64_t, DataStreamBlock>::iterator it;
                it = batch.blocks.find(blockNumber);
                if (it == batch.blocks.end())
                {
                    zklog.error("dataStream2batch() end L2 block, could not find existing blockNumber=" + to_string(blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check that the current block matches
                if (batch.currentBlock != it->second.blockNumber)
                {
                    zklog.error("dataStream2batch() end L2 block, found existing blockNumber=" + to_string(blockNumber) + " but currentblock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get new block data
                if (!it->second.l2BlockHash.empty())
                {
                    zklog.error("dataStream2batch() end L2 block, found l2BlockHash not empty, blockNumber=" + to_string(blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(it->second.l2BlockHash, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;
                if (!it->second.stateRoot.empty())
                {
                    zklog.error("dataStream2batch() end L2 block, found stateRoot not empty, blockNumber=" + to_string(blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(it->second.stateRoot, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;

#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() END L2 BLOCK " + it->second.toString());
#endif

                continue;
            }

            // Default: fail
            default:
            {
                zklog.error("dataStream2batch() unsupported entry type=" + to_string(type) + " data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                return ZKR_DATA_STREAM_INVALID_DATA;
            }
        }
    }

    // Check that batch is in the proper state, i.e. with current block closed
    if (batch.batchNumber == 0)
    {
        zklog.warning("dataStream2batch() final check, found batch.batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    if (batch.currentBlock == 0)
    {
        zklog.error("dataStream2batch() final check, found batch.currentBlock=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    map<uint64_t, DataStreamBlock>::iterator it;
    it = batch.blocks.find(batch.currentBlock);
    if (it == batch.blocks.end())
    {
        zklog.error("dataStream2batch() final check, could not find block batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    if (it->second.l2BlockHash.empty())
    {
        zklog.error("dataStream2batch() final check, found current block with l2BlockHash empty batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    if (it->second.stateRoot.empty())
    {
        zklog.error("dataStream2batch() final check, found current block with stateRoot empty batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }

    return ZKR_SUCCESS;
}

zkresult dataStreamBatch2batchL2Data (const DataStreamBatch &batch, string &batchL2Data)
{
    // Clear the result
    batchL2Data.clear();

    // For all blocks
    map<uint64_t, DataStreamBlock>::const_iterator it;
    for (it = batch.blocks.begin(); it != batch.blocks.end(); it++)
    {
        // Start of block
        batchL2Data.push_back(0x0b);

        // 4B = delta timestamp = timestamp - previous block timestamp (0 for block 0) -> new parameter?  If we had the whole DB we could get it from there and calculate the difference
        uint32_t deltaTimestamp = it->second.timestamp; // TODO: calculate deltaTimestamp
        batchL2Data += EncodeBigEndianU32(deltaTimestamp);

        //4B = L1 info tree index - It should be part of the datastream data, but not in the spec, yet
        uint32_t l1InfoTreeIndex = 0; // TODO: get it
        batchL2Data += EncodeBigEndianU32(l1InfoTreeIndex);

        const DataStreamBlock &block = it->second;
        for (uint64_t t = 0; t < block.txs.size(); t++)
        {
            const DataStreamTx &tx = block.txs[t];
            batchL2Data += tx.encodedTx; // TODO: Is rsv concatenated or part of the RLP itself?
            batchL2Data += tx.gasPricePercentage;
        }
    }

    return ZKR_SUCCESS;
}