#include "data_stream.hpp"
#include "zklog.hpp"
#include "scalar.hpp"
#include "rlp.hpp"
#include "grpc/gen/datastream.grpc.pb.h"
#include "timer.hpp"

//#define LOG_DATA_STREAM

uint8_t ParseU8 (const string &data, uint64_t &p)
{
    uint8_t result;
    result = data[p];
    p++;
    return result;
}

uint16_t ParseBigEndianU16 (const string &data, uint64_t &p)
{
    // Get the 2 bytes
    uint8_t d0 = data[p];
    uint8_t d1 = data[p+1];

    // Build the result
    uint32_t result;
    result = d0;
    result <<= 8;
    result += d1;

    // Increase the counter
    p += 2;

    return result;
}

uint32_t ParseBigEndianU32 (const string &data, uint64_t &p)
{
    // Get the 4 bytes
    uint8_t d0 = data[p];
    uint8_t d1 = data[p+1];
    uint8_t d2 = data[p+2];
    uint8_t d3 = data[p+3];

    // Build the result
    uint32_t result;
    result = d0;
    result <<= 8;
    result += d1;
    result <<= 8;
    result += d2;
    result <<= 8;
    result += d3;

    // Increase the counter
    p += 4;

    return result;
}

uint64_t ParseBigEndianU64 (const string &data, uint64_t &p)
{
    // Get the 8 bytes
    uint8_t d0 = data[p];
    uint8_t d1 = data[p+1];
    uint8_t d2 = data[p+2];
    uint8_t d3 = data[p+3];
    uint8_t d4 = data[p+4];
    uint8_t d5 = data[p+5];
    uint8_t d6 = data[p+6];
    uint8_t d7 = data[p+7];

    // Build the result
    uint32_t result;
    result = d0;
    result <<= 8;
    result += d1;
    result <<= 8;
    result += d2;
    result <<= 8;
    result += d3;
    result <<= 8;
    result += d4;
    result <<= 8;
    result += d5;
    result <<= 8;
    result += d6;
    result <<= 8;
    result += d7;

    // Increase the counter
    p += 8;

    return result;
}

string EncodeBigEndianU32 (uint32_t value)
{
    string result;
    const char * data = (const char *)&value;
    result.push_back(data[3]);
    result.push_back(data[2]);
    result.push_back(data[1]);
    result.push_back(data[0]);
    return result;
}

zkresult dataStream2batch (const string &dataStream, DataStreamBatch &batch)
{
    // Initialize variables
    uint64_t p = 0;
    batch.reset();
    string auxString;

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
        if (p + 4 > dataStream.size())
        {
            zklog.error("dataStream2batch() parsing length, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        uint32_t length = ParseBigEndianU32(dataStream, p);

        // Check length range
        if (length < 17)
        {
            zklog.error("dataStream2batch() checking length range, length=" + to_string(length) + "<17");
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        if (p + length - 1 /*packetType*/ - 4 /*length*/ - 1 > dataStream.size())
        {
            zklog.error("dataStream2batch() checking length, run out of data stream data p=" + to_string(p) + " length=" + to_string(length) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }

        // Parse type
        if (p + 4 > dataStream.size())
        {
            zklog.error("dataStream2batch() parsing type, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        uint32_t entryType = ParseBigEndianU32(dataStream, p);

        // Parse number
        if (p + 8 > dataStream.size())
        {
            zklog.error("dataStream2batch() parsing number, run out of data stream data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        //uint64_t number = ParseBigEndianU64(dataStream, p);
        p += 8;

        // Check that there is enough room for data
        uint64_t dataLength = length - 17;
        if (p + dataLength > dataStream.size())
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
        switch (entryType)
        {
            case 0xb0: // Bookmark type, skip
            {
                p += dataLength;
#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() BOOKMARK");
#endif
                continue;
            }

            case datastream::v1::ENTRY_TYPE_BATCH_START: // Batch start
            {
                // Check that batch has not been previously parsed
                if (batch.forkId != 0)
                {
                    zklog.error("dataStream2batch() batch start called with batch.forkId=" + to_string(batch.forkId) + " p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Parse the data stream entry as a protobuf
                datastream::v1::BatchStart dsBatchStart;
                if (!dsBatchStart.ParseFromString(dataStream.substr(p, dataLength)))
                {
                    zklog.error("dataStream2batch() batch start failed calling dsBatchStart.ParseFromString() p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                p += dataLength;

                // Get batch number
                uint64_t batchNumber = dsBatchStart.number();
                if (batchNumber == 0)
                {
                    zklog.error("dataStream2batch() batch start invalid batch number=0 p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                else
                {
                    batch.batchNumber = batchNumber;
                }

                // Get fork ID
                batch.forkId = dsBatchStart.fork_id();
                if (batch.forkId == 0)
                {
                    zklog.error("dataStream2batch() batch start invalid fork ID=0 p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get chain ID
                batch.chainId = dsBatchStart.chain_id();

#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() BATCH START " + batch.toString());
#endif

                continue;
            }

            case datastream::v1::ENTRY_TYPE_L2_BLOCK: // L2 block
            {
                // Check that batch has been previously parsed
                if (batch.forkId == 0)
                {
                    zklog.error("dataStream2batch() L2 block called with batch.forkId=0 p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Create a block and fill it with the entry data
                DataStreamBlock block;

                // Parse the data stream entry as a protobuf
                datastream::v1::L2Block dsBlock;
                if (!dsBlock.ParseFromString(dataStream.substr(p, dataLength)))
                {
                    zklog.error("dataStream2batch() L2 block failed calling dsBlock.ParseFromString() p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                p += dataLength;

                // Get block number
                block.blockNumber = dsBlock.number();
                if (block.blockNumber == 0)
                {
                    zklog.error("dataStream2batch() L2 block, found blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                
                // Get batch number
                uint64_t batchNumber;
                batchNumber = dsBlock.batch_number();
                if (batchNumber == 0)
                {
                    zklog.error("dataStream2batch() L2 block, found batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                if (batch.batchNumber == 0)
                {
                    batch.batchNumber = batchNumber;
                }
                else if (batchNumber != batch.batchNumber)
                {
                    zklog.error("dataStream2batch() L2 block, found batchNumber=" + to_string(batchNumber) + " != batch.batchNumber=" + to_string(batch.batchNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Parse block timestamp
                block.timestamp = dsBlock.timestamp();

                // Parse block delta timestamp
                block.deltaTimestamp = dsBlock.delta_timestamp();

                // Parse block min timestamp
                block.minTimestamp = dsBlock.min_timestamp();

                // Parse L1 block hash
                auxString = dsBlock.l1_blockhash();
                if (auxString.size() > 32)
                {
                    zklog.error("dataStream2batch() L2 block, found dsBlock.l1_blockhash.size=" + to_string(auxString.size()) + " != 32, blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(auxString, block.l1BlockHash);

                // Parse block L1 info tree index
                block.l1InfoTreeIndex = dsBlock.l1_infotree_index();

                // Parse L2 block hash
                auxString = dsBlock.hash();
                if (auxString.size() > 32)
                {
                    zklog.error("dataStream2batch() L2 block, found dsBlock.hash.size=" + to_string(auxString.size()) + " != 32, blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(auxString, block.l2BlockHash);

                // Parse state root
                auxString = dsBlock.state_root();
                if (auxString.size() > 32)
                {
                    zklog.error("dataStream2batch() L2 block, found dsBlock.state_root.size=" + to_string(auxString.size()) + " != 32, blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(auxString, block.stateRoot);

                // Parse block global exit root
                auxString = dsBlock.global_exit_root();
                if (auxString.size() > 32)
                {
                    zklog.error("dataStream2batch() L2 block, found dsBlock.global_exit_root.size=" + to_string(auxString.size()) + " != 32, blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(auxString, block.globalExitRoot);

                // Get coinbase
                auxString = dsBlock.coinbase();
                if (auxString.size() > 20)
                {
                    zklog.error("dataStream2batch() L2 block, found dsBlock.coinbase.size=" + to_string(auxString.size()) + " != 32, blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(auxString, block.coinbase);

                // If batch is empty, initialize it
                if (batch.blocks.empty())
                {
                    // Store block in batch
                    batch.blocks.emplace_back(block);
                }

                // If batch number has already been assigned, perform checks
                else
                {
                    // Check that the batch numbers match
                    if (batch.batchNumber != batchNumber) // If they don't match, we are getting blocks from different batches, so fail
                    {
                        zklog.error("dataStream2batch() L2 block, batch number mismatch, batchNumber=" + to_string(batchNumber) + " batch.batchNumber=" + to_string(batch.batchNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                        return ZKR_DATA_STREAM_INVALID_DATA;
                    }

                    // Check that the block number is incremental with regards the current one
                    uint64_t latestBlockNumber = batch.blocks[batch.blocks.size() - 1].blockNumber;
                    if (block.blockNumber != latestBlockNumber + 1)
                    {
                        zklog.error("dataStream2batch() L2 block, found block.blockNumber=" + to_string(block.blockNumber) + " different from 1 more than latestBlockNumber=" + to_string(latestBlockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                        return ZKR_DATA_STREAM_INVALID_DATA;
                    }

                    // Add the block to the batch list of blocks
                    batch.blocks.emplace_back(block);
                }

#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() L2 BLOCK " + block.toString());
#endif

                continue;
            }

            case datastream::v1::ENTRY_TYPE_TRANSACTION: // L2 TX
            {
                // Check that batch is in the proper state, i.e. with current block still open
                if (batch.blocks.empty())
                {
                    zklog.error("dataStream2batch() L2 TX found batch.blocks empty p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                DataStreamBlock &latestBlock = batch.blocks[batch.blocks.size() - 1];
                if (latestBlock.l2BlockHash.empty())
                {
                    zklog.error("dataStream2batch() L2 TX, found current block with l2BlockHash empty latestBlock=" + to_string(latestBlock.blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                if (latestBlock.stateRoot.empty())
                {
                    zklog.error("dataStream2batch() L2 TX, found current block with stateRoot empty latestBlock=" + to_string(latestBlock.blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Parse Tx
                DataStreamTx tx;
                
                // Parse the data stream entry as a protobuf
                datastream::v1::Transaction dsTx;
                if (!dsTx.ParseFromString(dataStream.substr(p, dataLength)))
                {
                    zklog.error("dataStream2batch() L2 TX failed calling dsTx.ParseFromString() p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                p += dataLength;

                // Check block number
                if (dsTx.l2block_number() != latestBlock.blockNumber)
                {
                    zklog.error("dataStream2batch() L2 TX invalid dsTx.l2block_number=" + to_string(dsTx.l2block_number()) + " != latestBlock.blockNumber=" + to_string(latestBlock.blockNumber) + " p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get is valid
                tx.isValid = dsTx.is_valid();

                // Get TX encoded data
                tx.encodedTx = dsTx.encoded();

                // Get gas price percentage
                tx.gasPricePercentage = dsTx.effective_gas_price_percentage();
                if (tx.gasPricePercentage > 255)
                {
                    zklog.error("dataStream2batch() L2 TX invalid tx.gasPricePercentage=" + to_string(tx.gasPricePercentage) + " p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get intermediate state root
                if (dsTx.im_state_root().size() > 32)
                {
                    zklog.error("dataStream2batch() L2 TX invalid dsTx.im_state_root=" + to_string(dsTx.im_state_root().size()) + " p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(dsTx.im_state_root(), tx.stateRoot);
                
                // Add it to the current block
                latestBlock.txs.emplace_back(tx);

#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() L2 TX " + tx.toString());
#endif

                continue;
            }

            case datastream::v1::ENTRY_TYPE_UPDATE_GER:
            {
                // Parse the update global exit root entry
                datastream::v1::UpdateGER dsUpdateGER;
                if (!dsUpdateGER.ParseFromString(dataStream.substr(p, dataLength)))
                {
                    zklog.error("dataStream2batch() Update GER failed calling dsUpdateGER.ParseFromString() p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                p += dataLength;

                // Check batch number
                if (dsUpdateGER.batch_number() != batch.batchNumber)
                {
                    zklog.error("dataStream2batch() Update GER found dsUpdateGER.batch_number=" + to_string(dsUpdateGER.batch_number()) + " != batch.batchNumber=" + to_string(batch.batchNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check timestamp
                /*if (dsUpdateGER.timestamp() != batch.timestamp)
                {
                    zklog.error("dataStream2batch() Update GER found dsUpdateGER.timestamp=" + to_string(dsUpdateGER.timestamp()) + " != batch.timestamp=" + to_string(batch.timestamp) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }*/

                // Check coinbase

                // Check fork ID
                if (dsUpdateGER.fork_id() != batch.forkId)
                {
                    zklog.error("dataStream2batch() Update GER found dsUpdateGER.fork_id=" + to_string(dsUpdateGER.fork_id()) + " != batch.forkId=" + to_string(batch.forkId) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check chain ID
                if (dsUpdateGER.chain_id() != batch.chainId)
                {
                    zklog.error("dataStream2batch() Update GER found dsUpdateGER.chain_id=" + to_string(dsUpdateGER.chain_id()) + " != batch.chainId=" + to_string(batch.chainId) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Check state root
                if (dsUpdateGER.state_root().size() > 32)
                {
                    zklog.error("dataStream2batch() Update GER invalid dsUpdateGER.state_root=" + to_string(dsUpdateGER.state_root().size()) + " p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                string stateRoot;
                ba2string(dsUpdateGER.state_root(), stateRoot);
                if (stateRoot != batch.stateRoot)
                {
                    zklog.error("dataStream2batch() Update GER found dsUpdateGER.state_root=" + stateRoot + " != batch.stateRoot=" + batch.stateRoot + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Update GER
                if (dsUpdateGER.global_exit_root().size() > 32)
                {
                    zklog.error("dataStream2batch() Update GER invalid dsUpdateGER.global_exit_root=" + to_string(dsUpdateGER.global_exit_root().size()) + " p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(dsUpdateGER.global_exit_root(), batch.globalExitRoot);
                
                continue;
            }

            case datastream::v1::ENTRY_TYPE_BATCH_END: // Batch end
            {
                // Check that batch has not been previously parsed
                if (batch.forkId == 0)
                {
                    zklog.error("dataStream2batch() batch end called with batch.forkId=0 p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Parse the data stream entry as a protobuf
                datastream::v1::BatchEnd dsBatchEnd;
                if (!dsBatchEnd.ParseFromString(dataStream.substr(p, dataLength)))
                {
                    zklog.error("dataStream2batch() batch end failed calling dsBatchStart.ParseFromString() p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                p += dataLength;

                // Get batch number
                uint64_t batchNumber = dsBatchEnd.number();
                if (batchNumber == 0)
                {
                    zklog.error("dataStream2batch() batch end invalid batch number=0 p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                else if (batchNumber != batch.batchNumber)
                {
                    zklog.error("dataStream2batch() batch end found batchNumber=" + to_string(batchNumber) + " != batch.batchNumber=" + to_string(batch.batchNumber) + " p=" + to_string(p) + " dataLength=" + to_string(dataLength) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }

                // Get local exit root
                auxString = dsBatchEnd.local_exit_root();
                if (auxString.size() > 32)
                {
                    zklog.error("dataStream2batch() batch end found dsBatch.local_exit_root.size=" + to_string(auxString.size()) + " != 32, blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(auxString, batch.localExitRoot);

                // Get state root
                auxString = dsBatchEnd.state_root();
                if (auxString.size() > 32)
                {
                    zklog.error("dataStream2batch() batch end found dsBatch.state_root.size=" + to_string(auxString.size()) + " != 32, blockNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                    return ZKR_DATA_STREAM_INVALID_DATA;
                }
                ba2string(auxString, batch.stateRoot);

#ifdef LOG_DATA_STREAM
                zklog.info("dataStream2batch() BATCH END " + batch.toString());
#endif

                continue;
            }

            case datastream::v1::ENTRY_TYPE_L2_BLOCK_END: // L2 block end
            {
                // Ignore, just pretend to have parsed this data
                p += dataLength;

                continue;
            }

            // Default: fail
            default:
            {
                zklog.error("dataStream2batch() unsupported entry type=" + to_string(entryType) + " data p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
                return ZKR_DATA_STREAM_INVALID_DATA;
            }
        }
    }

    // Check that batch is in the proper state, i.e. with latest block closed
    if (!batch.blocks.empty())
    {
        if (batch.batchNumber == 0)
        {
            zklog.warning("dataStream2batch() final check, found batch.batchNumber=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        if (batch.forkId == 0)
        {
            zklog.error("dataStream2batch() final check, found batch.forkId=0 p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        DataStreamBlock &latestBlock = batch.blocks[batch.blocks.size() - 1];
        if (latestBlock.l2BlockHash.empty())
        {
            zklog.error("dataStream2batch() final check, found current block with l2BlockHash empty latestBlock.blockNumber=" + to_string(latestBlock.blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
        if (latestBlock.stateRoot.empty())
        {
            zklog.error("dataStream2batch() final check, found current block with stateRoot empty latestBlock.blockNumber=" + to_string(latestBlock.blockNumber) + " p=" + to_string(p) + " dataStream.size=" + to_string(dataStream.size()));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }
    }

#ifdef LOG_DATA_STREAM_BATCH
    zklog.info("dataStream2batch() got:");
    string log = batch.toString() + "\n";
    for (uint64_t b=0; b<batch.blocks.size(); b++)
    {
        log += "  blocks[" + to_string(b) + "]= " + batch.blocks[b].toString() + "\n";
        for (uint64_t t=0; t<batch.blocks[b].txs.size(); t++)
        {
            log += "    txs[" + to_string(t) + "]= " + batch.blocks[b].txs[t].toString() + " encodedTx=" + ba2string(batch.blocks[b].txs[t].encodedTx) + "\n";
        }
    }
    cout << log << endl;
#endif

    return ZKR_SUCCESS;
}

zkresult dataStreamBatch2batchL2Data (const DataStreamBatch &batch, string &batchL2Data)
{
    // Clear the result
    batchL2Data.clear();

    // For all blocks
    for (uint64_t b=0; b < batch.blocks.size(); b++)
    {
        const DataStreamBlock &block = batch.blocks[b];

        // Start of block
        batchL2Data.push_back(0x0b);

        // 4B = delta timestamp = timestamp - previous block timestamp (0 for block 0)
        uint32_t deltaTimestamp = block.deltaTimestamp;
        batchL2Data += EncodeBigEndianU32(deltaTimestamp);

        //4B = L1 info tree index 
        uint32_t l1InfoTreeIndex = block.l1InfoTreeIndex;
        batchL2Data += EncodeBigEndianU32(l1InfoTreeIndex);

        for (uint64_t t = 0; t < block.txs.size(); t++)
        {
            const DataStreamTx &tx = block.txs[t];
            string transcodedTx;
            zkresult zkr = transcodeTx(tx.encodedTx, batch.chainId, transcodedTx);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("dataStreamBatch2batchL2Data() failed calling transcodeTx() result=" + zkresult2string(zkr));
                return zkr;
            }
            batchL2Data += transcodedTx;
            batchL2Data += tx.gasPricePercentage;
        }
    }

    //zklog.info("dataStreamBatch2batchL2Data() generated data of size=" + to_string(batchL2Data.size()));

    return ZKR_SUCCESS;
}

zkresult dataStream2batchL2Data (const string &dataStream, DataStreamBatch &batch, string &batchL2Data)
{
    // Save start time
    struct timeval t;
    gettimeofday(&t, NULL);

    // Call dataStream2batch
    zkresult zkr = dataStream2batch(dataStream, batch);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("dataStream2batchL2Data() failed calling dataStream2batch() zkr=" + zkresult2string(zkr));
        return zkr;
    }

    // Call dataStreamBatch2batchL2Data
    zkr = dataStreamBatch2batchL2Data(batch, batchL2Data);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("dataStream2batchL2Data() failed calling dataStreamBatch2batchL2Data() zkr=" + zkresult2string(zkr));
        return zkr;
    }

    zklog.info("dataStream2batchL2Data() done dataStream.size=" + to_string(dataStream.size()) + " batch=" + batch.toString() + " batchL2Data.size=" + to_string(batchL2Data.size()) + " in " + to_string(TimeDiff(t)) + "us");
    
    return zkr;
}

//#define LOG_TX_FIELDS

// Decodes tx from Ethereum RLP format, and encodes it into ROM RLP format
// From: RLP(fields, v, r, s) --> To: RLP(fields, chainId, 0, 0) | r | s | v
zkresult transcodeTx (const string &tx, uint32_t batchChainId, string &transcodedTx)
{
    //zklog.info("tx original=" + ba2string(tx));
    
    // Decode the TX RLP list
    bool bResult;
    vector<string> fields;
    bResult = rlp::decodeList(tx, fields);
    if (!bResult)
    {
        zklog.error("transcodeTx() failed calling decodeList()");
        return ZKR_DATA_STREAM_INVALID_DATA;
    }

    // Expected TX fields:
    // 0: nonce
    // 1: gas price
    // 2: gas limit
    // 3: to
    // 4: value
    // 5: data
    // 6: v --> We can get the chain ID from tx.v
    // 7: r
    // 8: s

    // Check fields size
    if (fields.size() != 9)
    {
        zklog.error("transcodeTx() called decodeList() and got invalid fields.size=" + to_string(fields.size()));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }

#ifdef LOG_TX_FIELDS
    mpz_class auxScalar;
    ba2scalar(auxScalar, fields[0]);
    zklog.info("TX nonce=" + auxScalar.get_str(10));
    ba2scalar(auxScalar, fields[1]);
    zklog.info("TX gas price=" + auxScalar.get_str(10));
    ba2scalar(auxScalar, fields[2]);
    zklog.info("TX gas limit=" + auxScalar.get_str(10));
    ba2scalar(auxScalar, fields[3]);
    zklog.info("TX to=" + auxScalar.get_str(16));
    ba2scalar(auxScalar, fields[4]);
    zklog.info("TX value=" + auxScalar.get_str(10));
#endif

    // Get TX v
    mpz_class vScalar;
    ba2scalar(vScalar, fields[6]);
    if (vScalar > ScalarMask64)
    {
        zklog.error("transcodeTx() called decodeList() and got too big v=" + vScalar.get_str(16));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    uint64_t txv = vScalar.get_ui();

#ifdef LOG_TX_FIELDS
    zklog.info("TX v original=" + to_string(txv));
#endif
    
    // Get ROM v
    uint64_t v;
    bool isPreEIP155;
    if ((txv == 27) || (txv == 28)) // This is a pre-EIP-155
    {
        isPreEIP155 = true;

        v = txv;

#ifdef LOG_TX_FIELDS
    zklog.info("TX pre-EIP-155 v rom=" + to_string(v));
#endif

    }
    else
    {
        isPreEIP155 = false;

        // Get chain ID
        uint64_t chainId = (txv - 35) / 2;
        if (chainId != batchChainId)
        {
            zklog.error("transcodeTx() called decodeList() and got chainId=" + to_string(chainId) + " != batchChainId=" + to_string(batchChainId));
            return ZKR_DATA_STREAM_INVALID_DATA;
        }

        // Get ROM v
        v = txv - chainId*2 - 35 + 27;

#ifdef LOG_TX_FIELDS
    zklog.info("TX EIP-155 v rom=" + to_string(v) + " chainID=" + to_string(chainId));
#endif
    }

    // Get r
    mpz_class r;
    ba2scalar(r, fields[7]);
    if (r > ScalarMask256)
    {
        zklog.error("transcodeTx() called decodeList() and got too big r=" + r.get_str(16));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    
#ifdef LOG_TX_FIELDS
    zklog.info("TX r =" + r.get_str(16) + "=" + r.get_str(10));
#endif

    // Get s
    mpz_class s;
    ba2scalar(s, fields[8]);
    if (s > ScalarMask256)
    {
        zklog.error("transcodeTx() called decodeList() and got too big r=" + r.get_str(16));
        return ZKR_DATA_STREAM_INVALID_DATA;
    }
    
#ifdef LOG_TX_FIELDS
    zklog.info("TX s =" + s.get_str(16) + "=" + s.get_str(10));
#endif

    // preEIP155 format: [rlp(nonce,gasprice,gaslimit,to,value,data)|r|s|v|effectivePercentage]
    if (isPreEIP155)
    {
        fields.pop_back();
        fields.pop_back();
        fields.pop_back();
    }
    // Legacy format: [rlp(nonce,gasprice,gaslimit,to,value,data,chainId,0,0)|r|s|v|effectivePercentage]
    else
    {
        // Set fields[6] = chain ID
        fields[6].clear();
        const uint8_t * pChainId = (const uint8_t *)&batchChainId;
        bool writing = false;
        for (int64_t i = 3; i >= 0; i--)
        {
            if (writing || (pChainId[i] != 0))
            {
                fields[6] += pChainId[i];
                writing = true;
            }
        }
        
#ifdef LOG_TX_FIELDS
        zklog.info("TX chain ID=" + to_string(batchChainId));
#endif

        // Clear fields[7]
        fields[7].clear();

        // Clear fields[8]
        fields[8].clear();
    }

    // Encode RLP list
    bResult = rlp::encodeList(fields, transcodedTx);
    if (!bResult)
    {
        zklog.error("transcodeTx() failed calling encodeList()");
        return ZKR_DATA_STREAM_INVALID_DATA;
    }

    // Format r and concatenate to tx
    string rBa;
    rBa = scalar2ba32(r);
    transcodedTx += rBa;
    //zklog.info("r=" + ba2string(rBa));

    // Format s and concatenate to tx
    string sBa;
    sBa = scalar2ba32(s);
    transcodedTx += sBa;
    //zklog.info("s=" + ba2string(sBa));

    // Concatenat v to tx
    uint8_t d = v;
    transcodedTx += d;
    //zklog.info("v=" + to_string(v));

    //zklog.info("transcoded tx=" + ba2string(transcodedTx));

    return ZKR_SUCCESS;
}