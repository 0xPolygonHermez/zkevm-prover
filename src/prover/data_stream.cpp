#include "data_stream.hpp"
#include "zklog.hpp"
#include "scalar.hpp"

zkresult dataStream2data (const string &dataStream, DataStreamBatch &batch)
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
        uint8_t packetType = dataStream[p];
        p++;

        // Parse length
        if (p + 3 >= dataStream.size())
        {
            zklog.error("dataStream2data() parsing length, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        uint32_t length = *(&dataStream[p]);
        p += 4;

        // Check length range
        if (length < 17)
        {
            zklog.error("dataStream2data() checking length range, length=" + to_string(length) + "<17");
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        if (p + length - 1 /*packetType*/ - 4 /*length*/ < dataStream.size())
        {
            zklog.error("dataStream2data() checking length, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }

        // Parse type
        if (p + 3 >= dataStream.size())
        {
            zklog.error("dataStream2data() parsing type, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        uint32_t type = *(&dataStream[p]);
        p += 4;

        // Parse number
        if (p + 3 >= dataStream.size())
        {
            zklog.error("dataStream2data() parsing number, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        //uint32_t number = *(&dataStream[p]);
        p += 4;

        // Check that there is enough room for data
        uint64_t dataLength = length - 17;
        if (p + dataLength - 1 >= dataStream.size())
        {
            zklog.error("dataStream2data() checking data length, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
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
            zklog.error("dataStream2data() unsupported packet type=" + to_string(p) + " data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }

        // Check type
        switch (type)
        {
            case 0xb0: // Bookmark type, skip
            {
                p += dataLength;
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
                    zklog.error("dataStream2data() start L2 block invalid dataLength=" + to_string(dataLength) + "!=110 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get batch number
                uint64_t batchNumber;
                batchNumber = *(uint64_t *)(&dataStream[p]);
                p += 8;

                // Check it is not zero
                if (batchNumber == 0)
                {
                    zklog.error("dataStream2data() start L2 block, found batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Create a block and fill it with the entry data
                DataStreamBlock block;
                block.blockNumber = *(uint64_t *)(&dataStream[p]);
                p += 8;
                if (block.blockNumber == 0)
                {
                    zklog.error("dataStream2data() end L2 block, found blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                block.timestamp = *(uint64_t *)(&dataStream[p]);
                p += 8;
                ba2string(block.l1BlockHash, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;
                ba2string(block.globalExitRoot, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;
                ba2string(block.coinbase, (const uint8_t *)dataStream.c_str() + p, 20);
                p += 20;
                uint16_t forkId;
                forkId = *(uint16_t *)(&dataStream[p]);
                p += 2;
                block.forkId = forkId;

                // If batch number has already been assigned
                if (batch.batchNumber != 0)
                {
                    // Check that the batch numbers match
                    if (batch.batchNumber != batchNumber) // If they don't match, we are getting blocks from different batches, so fail
                    {
                        zklog.error("dataStream2data() start L2 block, batch number mismatch, batchNumber=" + to_string(batchNumber) + " batch.batchNumber=" + to_string(batch.batchNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                        return ZKR_DATA_STREAM_INVALID_DATA;
                    }
                    
                    // If batch number was not zero, check the block did not exist previously
                    map<uint64_t, DataStreamBlock>::iterator it;
                    it = batch.blocks.find(block.blockNumber);
                    if (it != batch.blocks.end())
                    {
                        zklog.error("dataStream2data() start L2 block, found existing block number=" + to_string(block.blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                        return ZKR_DATA_STREAM_INVALID_DATA;
                    }
                }
                else
                {
                    // Record the first batch number found
                    batch.batchNumber = batchNumber;
                }

                // Store block in batch
                batch.blocks[block.blockNumber] = block;
                batch.currentBlock = block.blockNumber;

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
                    zklog.error("dataStream2data() L2 TX invalid dataLength=" + to_string(dataLength) + "<38 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check that batch is in the proper state, i.e. with current block still open
                if (batch.batchNumber == 0)
                {
                    zklog.error("dataStream2data() L2 TX found batch.batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                if (batch.currentBlock == 0)
                {
                    zklog.error("dataStream2data() L2 TX found batch.currentBlock=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                map<uint64_t, DataStreamBlock>::iterator it;
                it = batch.blocks.find(batch.currentBlock);
                if (it == batch.blocks.end())
                {
                    zklog.error("dataStream2data() L2 TX, could not find block batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                if (!it->second.l2BlockHash.empty())
                {
                    zklog.error("dataStream2data() L2 TX, found current block with l2BlockHash not empty batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                if (!it->second.stateRoot.empty())
                {
                    zklog.error("dataStream2data() L2 TX, found current block with stateRoot not empty batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Parse Tx
                DataStreamTx tx;
                tx.gasPricePercentage = *(uint8_t *)(&dataStream[p]);
                p += 1;
                tx.isValid = *(uint8_t *)(&dataStream[p]);
                p += 1;
                ba2string(tx.stateRoot, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;
                uint32_t encodedTxLength = *(uint32_t *)(&dataStream[p]);
                p += 4;
                if (p + encodedTxLength - 1 >= dataStream.size())
                {
                    zklog.error("dataStream2data() L2 TX, run out of data batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                tx.encodedTx = dataStream.substr(p, encodedTxLength);
                p += encodedTxLength;

                // Add it to the current block
                it->second.txs.emplace_back(tx);

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
                    zklog.error("dataStream2data() end L2 block invalid dataLength=" + to_string(dataLength) + "!=72 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get block number
                uint64_t blockNumber;
                blockNumber = *(uint64_t *)(&dataStream[p]);
                p += 8;

                // Check it is not zero
                if (blockNumber == 0)
                {
                    zklog.error("dataStream2data() end L2 block, found blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check that batch number has already been assigned
                if (batch.batchNumber == 0)
                {
                    zklog.error("dataStream2data() end L2 block, found batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                    
                // Check the block did exist previously
                map<uint64_t, DataStreamBlock>::iterator it;
                it = batch.blocks.find(blockNumber);
                if (it == batch.blocks.end())
                {
                    zklog.error("dataStream2data() end L2 block, could not find existing blockNumber=" + to_string(blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check that the current block matches
                if (batch.currentBlock != it->second.blockNumber)
                {
                    zklog.error("dataStream2data() end L2 block, found existing blockNumber=" + to_string(blockNumber) + " but currentblock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get new block data
                if (!it->second.l2BlockHash.empty())
                {
                    zklog.error("dataStream2data() end L2 block, found l2BlockHash not empty, blockNumber=" + to_string(blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(it->second.l2BlockHash, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;
                if (!it->second.stateRoot.empty())
                {
                    zklog.error("dataStream2data() end L2 block, found stateRoot not empty, blockNumber=" + to_string(blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(it->second.stateRoot, (const uint8_t *)dataStream.c_str() + p, 32);
                p += 32;

                continue;
            }

            // Default: fail
            default:
            {
                zklog.error("dataStream2data() unsupported entry type=" + to_string(type) + " data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                return ZKR_DATA_STREAM_INVALID_DATA;
            }
        }
    }

    // Check that batch is in the proper state, i.e. with current block closed
    if (batch.batchNumber == 0)
    {
        zklog.warning("dataStream2data() final check, found batch.batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    if (batch.currentBlock == 0)
    {
        zklog.error("dataStream2data() final check, found batch.currentBlock=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    map<uint64_t, DataStreamBlock>::iterator it;
    it = batch.blocks.find(batch.currentBlock);
    if (it == batch.blocks.end())
    {
        zklog.error("dataStream2data() final check, could not find block batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    if (it->second.l2BlockHash.empty())
    {
        zklog.error("dataStream2data() final check, found current block with l2BlockHash empty batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    if (it->second.stateRoot.empty())
    {
        zklog.error("dataStream2data() final check, found current block with stateRoot empty batch.currentBlock=" + to_string(batch.currentBlock) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }

    return ZKR_SUCCESS;
}
