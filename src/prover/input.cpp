#include <iostream>
#include "config.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "database.hpp"
#include "utils.hpp"
#include "zklog.hpp"

zkresult Input::load (json &input)
{
    loadGlobals(input);
    loadDatabase(input);

    return ZKR_SUCCESS;
}

void Input::save (json &input) const
{
    saveGlobals(input);
    saveDatabase(input);
}

void Input::save (json &input, DatabaseMap &dbReadLog) const
{
    saveGlobals(input);
    saveDatabase(input, dbReadLog);
}

/* Load old/new state roots, sequencer address and chain ID */

void Input::loadGlobals (json &input)
{
    string auxString;

    /*****************/
    /* PUBLIC INPUTS */
    /*****************/

    // Input JSON could contain a dataStream key
    if (input.contains("dataStream") && input["dataStream"].is_string())
    {
        string dataStream = input["dataStream"];
        publicInputsExtended.publicInputs.dataStream = string2ba(dataStream);
    }

    // Input JSON file could contain a forkID key at the root level (not mandatory, default is 0)
    if ( input.contains("forkID") &&
         input["forkID"].is_number_unsigned() )
    {
        publicInputsExtended.publicInputs.forkID = input["forkID"];
    }
#ifdef LOG_INPUT
    zklog.info("Input::loadGlobals(): forkID=" + to_string(publicInputsExtended.publicInputs.forkID));
#endif

    // Input JSON file might contain a oldStateRoot key at the root level
    if ( input.contains("oldStateRoot") &&
         input["oldStateRoot"].is_string() )
    {
        auxString = Remove0xIfPresent(input["oldStateRoot"]);
        publicInputsExtended.publicInputs.oldStateRoot.set_str(auxString, 16);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): oldStateRoot=" + publicInputsExtended.publicInputs.oldStateRoot.get_str(16));
#endif
    }

    // Input JSON file may contain a oldAccInputHash key at the root level
    if ( input.contains("oldAccInputHash") &&
         input["oldAccInputHash"].is_string() )
    {
        auxString = Remove0xIfPresent(input["oldAccInputHash"]);
        publicInputsExtended.publicInputs.oldAccInputHash.set_str(auxString, 16);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): oldAccInputHash=" + publicInputsExtended.publicInputs.oldAccInputHash.get_str(16));
#endif
    }

    // Input JSON file must contain a oldBatchNum key at the root level
    if ( input.contains("oldNumBatch") &&
         input["oldNumBatch"].is_number_unsigned() )
    {
        publicInputsExtended.publicInputs.oldBatchNum = input["oldNumBatch"];
#ifdef LOG_INPUT
        zklog.info("loadGlobals(): oldBatchNum=" + to_string(publicInputsExtended.publicInputs.oldBatchNum));
#endif
    }

    // Input JSON file must contain a chainID key at the root level (it is mandatory)
    if ( !input.contains("chainID") ||
         !input["chainID"].is_number_unsigned() )
    {
        zklog.error("Input::loadGlobals() chainID key not found in input JSON file");
        exitProcess();
    }
    else
    {
        publicInputsExtended.publicInputs.chainID = input["chainID"];
    }
#ifdef LOG_INPUT
    zklog.info("Input::loadGlobals(): chainID=" + to_string(publicInputsExtended.publicInputs.chainID));
#endif

    // Input JSON file might contain a batchL2Data key at the root level
    if ( input.contains("batchL2Data") &&
         input["batchL2Data"].is_string() )
    {
        publicInputsExtended.publicInputs.batchL2Data = string2ba(input["batchL2Data"]);
    }

    // Check the batchL2Data length
    if (publicInputsExtended.publicInputs.batchL2Data.size() > (MAX_BATCH_L2_DATA_SIZE))
    {
        zklog.error("Input::loadGlobals() found batchL2Data.size()=" + to_string(publicInputsExtended.publicInputs.batchL2Data.size()) + " > MAX_BATCH_L2_DATA_SIZE=" + to_string(MAX_BATCH_L2_DATA_SIZE));
        exitProcess();
    }
#ifdef LOG_INPUT
    zklog.info("loadGlobals(): batchL2Data=" + ba2string(publicInputsExtended.publicInputs.batchL2Data));
#endif

    if ((publicInputsExtended.publicInputs.forkID >= 1) && (publicInputsExtended.publicInputs.forkID <= 6))
    {
        // Input JSON file must contain a globalExitRoot key at the root level
        if ( !input.contains("globalExitRoot") ||
            !input["globalExitRoot"].is_string() )
        {
            zklog.error("Input::loadGlobals() globalExitRoot key not found in input JSON file");
            exitProcess();
        }
        publicInputsExtended.publicInputs.globalExitRoot.set_str(Remove0xIfPresent(input["globalExitRoot"]), 16);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): globalExitRoot=" + publicInputsExtended.publicInputs.globalExitRoot.get_str(16));
#endif
    }

    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        // Input JSON file must contain a l1InfoRoot key at the root level
        if ( !input.contains("l1InfoRoot") ||
            !input["l1InfoRoot"].is_string() )
        {
            zklog.error("Input::loadGlobals() l1InfoRoot key not found in input JSON file");
            exitProcess();
        }
        publicInputsExtended.publicInputs.l1InfoRoot.set_str(Remove0xIfPresent(input["l1InfoRoot"]), 16);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): l1InfoRoot=" + publicInputsExtended.publicInputs.l1InfoRoot.get_str(16));
#endif
    }

    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        // Input JSON file must contain a forcedBlockHashL1 key at the root level
        if ( !input.contains("forcedBlockHashL1") ||
            !input["forcedBlockHashL1"].is_string() )
        {
            zklog.error("Input::loadGlobals() forcedBlockHashL1 key not found in input JSON file");
            exitProcess();
        }
        publicInputsExtended.publicInputs.forcedBlockHashL1.set_str(Remove0xIfPresent(input["forcedBlockHashL1"]), 16);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): forcedBlockHashL1=" + publicInputsExtended.publicInputs.forcedBlockHashL1.get_str(16));
#endif
    }

    if ((publicInputsExtended.publicInputs.forkID >= 1) && (publicInputsExtended.publicInputs.forkID <= 6))
    {
        // Input JSON file must contain a timestamp key at the root level
        if ( !input.contains("timestamp") ||
            !input["timestamp"].is_number_unsigned() )
        {
            zklog.error("Input::loadGlobals() timestamp key not found in input JSON file");
            exitProcess();
        }
        publicInputsExtended.publicInputs.timestamp = input["timestamp"];
#ifdef LOG_INPUT
        zklog.info("loadGlobals(): timestamp=" + to_string(publicInputsExtended.publicInputs.timestamp));
#endif
    }

    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        // Input JSON file must contain a timestampLimit key at the root level
        if ( !input.contains("timestampLimit") )
        {
            zklog.error("Input::loadGlobals() timestampLimit key not found in input JSON file");
            exitProcess();
        }
        // Parse it based on its type
        if ( input["timestampLimit"].is_number_unsigned() )
        {
            publicInputsExtended.publicInputs.timestampLimit = input["timestampLimit"];
        }
        else if ( input["timestampLimit"].is_string() )
        {
            string timestampLimitString = input["timestampLimit"];
            if (stringIsDec(timestampLimitString))
            {
                mpz_class timestampLimitScalar;
                timestampLimitScalar.set_str(timestampLimitString, 10);
                if (timestampLimitScalar > ScalarMask64)
                {
                    zklog.error("Input::loadGlobals() timestampLimit key found in input JSON file is too big value=" + timestampLimitString);
                    exitProcess();
                }
                publicInputsExtended.publicInputs.timestampLimit = timestampLimitScalar.get_ui();
            }
            else
            {
                timestampLimitString = Remove0xIfPresent(timestampLimitString);
                if (stringIsHex(timestampLimitString))
                {
                    mpz_class timestampLimitScalar;
                    timestampLimitScalar.set_str(timestampLimitString, 16);
                    if (timestampLimitScalar > ScalarMask64)
                    {
                        zklog.error("Input::loadGlobals() timestampLimit key found in input JSON file is too big value=" + timestampLimitString);
                        exitProcess();
                    }
                    publicInputsExtended.publicInputs.timestampLimit = timestampLimitScalar.get_ui();
                }
                else
                {
                    zklog.error("Input::loadGlobals() timestampLimit key found in input JSON file is not decimal nor hexa value=" + timestampLimitString);
                    exitProcess();
                }
            }
        }
        else
        {
            zklog.error("Input::loadGlobals() timestampLimit invalid type");
            exitProcess();
        }

#ifdef LOG_INPUT
        zklog.info("loadGlobals(): timestampLimit=" + to_string(publicInputsExtended.publicInputs.timestampLimit));
#endif
    }

    // Input JSON file must contain a sequencerAddr key at the root level
    if ( !input.contains("sequencerAddr") ||
         !input["sequencerAddr"].is_string() )
    {
        zklog.error("Input::loadGlobals() sequencerAddr key not found in input JSON file");
        exitProcess();
    }
    publicInputsExtended.publicInputs.sequencerAddr.set_str(Remove0xIfPresent(input["sequencerAddr"]), 16);
#ifdef LOG_INPUT
    zklog.info("Input::loadGlobals(): sequencerAddr=" + publicInputsExtended.publicInputs.sequencerAddr.get_str(16));
#endif

    // Input JSON file may contain a aggregatorAddress key at the root level
    if ( input.contains("aggregatorAddress") &&
         input["aggregatorAddress"].is_string() )
    {
        auxString = Remove0xIfPresent(input["aggregatorAddress"]);
        publicInputsExtended.publicInputs.aggregatorAddress.set_str(auxString, 16);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): aggregatorAddress=" + publicInputsExtended.publicInputs.aggregatorAddress.get_str(16));
#endif
    }

    /**************************/
    /* PUBLIC INPUTS EXTENDED */
    /**************************/

    // Input JSON file may contain a newStateRoot key at the root level
    if ( input.contains("newStateRoot") &&
         input["newStateRoot"].is_string() )
    {
        auxString = Remove0xIfPresent(input["newStateRoot"]);
        publicInputsExtended.newStateRoot.set_str(auxString, 16);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): newStateRoot=" + publicInputsExtended.newStateRoot.get_str(16));
#endif
    }

    // Input JSON file may contain a newAccInputHash key at the root level
    if ( input.contains("newAccInputHash") &&
         input["newAccInputHash"].is_string() )
    {
        auxString = Remove0xIfPresent(input["newAccInputHash"]);
        publicInputsExtended.newAccInputHash.set_str(auxString, 16);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() newAccInputHash=" + publicInputsExtended.newAccInputHash.get_str(16));
#endif
    }

    // Input JSON file may contain a newLocalExitRoot key at the root level
    if ( input.contains("newLocalExitRoot") &&
         input["newLocalExitRoot"].is_string() )
    {
        auxString = Remove0xIfPresent(input["newLocalExitRoot"]);
        publicInputsExtended.newLocalExitRoot.set_str(auxString, 16);

#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): newLocalExitRoot=" + publicInputsExtended.newLocalExitRoot.get_str(16));
#endif
    }

    // Input JSON file may contain a numBatch key at the root level
    if ( input.contains("newNumBatch") &&
         input["newNumBatch"].is_number_unsigned() )
    {
        publicInputsExtended.newBatchNum = input["newNumBatch"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): newBatchNum=" + to_string(publicInputsExtended.newBatchNum));
#endif
    }

    /********/
    /* ROOT */
    /********/

    // Input JSON file may contain a from key at the root level
    if ( input.contains("from") &&
         input["from"].is_string() )
    {
        from = Add0xIfMissing(input["from"]);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() from=" + from);
#endif
    }

    // Input JSON file may contain an updateMerkleTree key at the root level
    if ( input.contains("updateMerkleTree") &&
         input["updateMerkleTree"].is_boolean() )
    {
        bUpdateMerkleTree = input["updateMerkleTree"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() updateMerkleTree=" + to_string(bUpdateMerkleTree));
#endif
    }

    // Input JSON file may contain a noCounters key at the root level
    if ( input.contains("noCounters") &&
         input["noCounters"].is_boolean() )
    {
        bNoCounters = input["noCounters"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() noCounters=" + to_string(bNoCounters));
#endif
    }

    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        // Input JSON file might contain a stepsN key at the root level
        if ( input.contains("stepsN") &&
            input["stepsN"].is_number_unsigned() )
        {
            stepsN = input["stepsN"];
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): stepsN=" + to_string(stepsN));
#endif
        }
    }

    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        // Input JSON file might contain a gasLimit key at the root level
        if ( input.contains("gasLimit") &&
            input["gasLimit"].is_string() )
        {
            string gasLimitString = input["gasLimit"];
            mpz_class gasLimitScalar;
            gasLimitScalar.set_str(gasLimitString, 10);
            if (gasLimitScalar > ScalarMask64)
            {
                zklog.error("Input::loadGlobals() gasLimit key value is too high value=" + gasLimitString);
                exitProcess();
            }
            debug.gasLimit = gasLimitScalar.get_ui();
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): gasLimit=" + to_string(debug.gasLimit));
#endif
        }
    }

    // Input JSON file may contain a getKeys key at the root level
    if ( input.contains("getKeys") &&
         input["getKeys"].is_boolean() )
    {
        bGetKeys = input["getKeys"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() bGetKeys=" + to_string(bGetKeys));
#endif
    }

    // Input JSON file may contain a skipVerifyL1InfoRoot key at the root level
    if ( input.contains("skipVerifyL1InfoRoot") &&
         input["skipVerifyL1InfoRoot"].is_boolean() )
    {
        bSkipVerifyL1InfoRoot = input["skipVerifyL1InfoRoot"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() bSkipVerifyL1InfoRoot=" + to_string(bSkipVerifyL1InfoRoot));
#endif
    }

    // Input JSON file may contain a skipFirstChangeL2Block key at the root level
    if ( input.contains("skipFirstChangeL2Block") &&
         input["skipFirstChangeL2Block"].is_boolean() )
    {
        bSkipFirstChangeL2Block = input["skipFirstChangeL2Block"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() bSkipFirstChangeL2Block=" + to_string(bSkipFirstChangeL2Block));
#endif
    }

    // Input JSON file may contain a skipWriteBlockInfoRoot key at the root level
    if ( input.contains("skipWriteBlockInfoRoot") &&
         input["skipWriteBlockInfoRoot"].is_boolean() )
    {
        bSkipWriteBlockInfoRoot = input["skipWriteBlockInfoRoot"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() bSkipWriteBlockInfoRoot=" + to_string(bSkipWriteBlockInfoRoot));
#endif
    }

    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        // Input JSON file can contain a l1InfoTree key at the root level
        if ( input.contains("l1InfoTree") &&
             input["l1InfoTree"].is_object() )
        {
            /*if ( input["l1InfoTree"].contains("skipVerifyL1InfoRoot") &&
                 input["l1InfoTree"]["skipVerifyL1InfoRoot"].is_boolean() )
            {
                bSkipVerifyL1InfoRoot = input["l1InfoTree"]["skipVerifyL1InfoRoot"];
            }*/
            json::iterator it;
            for (it = input["l1InfoTree"].begin(); it != input["l1InfoTree"].end(); it++)
            {
                string key = it.key();
                if (key == "skipVerifyL1InfoRoot")
                {
                    if (it.value().is_boolean())
                    {
                        bSkipVerifyL1InfoRoot = input["l1InfoTree"]["skipVerifyL1InfoRoot"];
                    }
                    else
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree skipVerifyL1InfoRoot found in input JSON file but with invalid type");
                        exitProcess();
                    }
                    continue;
                }
                
                L1Data l1Data;

                if (!stringIsDec(key))
                {
                    continue;
                }
                uint64_t index = atoi(key.c_str());

                // Parse global exit root
                if ( input["l1InfoTree"][key].contains("globalExitRoot") &&
                     input["l1InfoTree"][key]["globalExitRoot"].is_string() )
                {
                    string globalExitRootString = input["l1InfoTree"][key]["globalExitRoot"];
                    globalExitRootString = Remove0xIfPresent(globalExitRootString);
                    if (!stringIsHex(globalExitRootString))
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree globalExitRoot found in input JSON file but not an hex string");
                        exitProcess();
                    }
                    if (globalExitRootString.size() > 64)
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree globalExitRoot found in input JSON file is too long");
                        exitProcess();
                    }
                    l1Data.globalExitRoot.set_str(globalExitRootString, 16);
                }

                // Parse block hash
                if ( input["l1InfoTree"][key].contains("blockHash") &&
                     input["l1InfoTree"][key]["blockHash"].is_string() )
                {
                    string blockHashString = input["l1InfoTree"][key]["blockHash"];
                    blockHashString = Remove0xIfPresent(blockHashString);
                    if (!stringIsHex(blockHashString))
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree blockHash found in input JSON file but not an hex string");
                        exitProcess();
                    }
                    if (blockHashString.size() > 64)
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree blockHash found in input JSON file is too long");
                        exitProcess();
                    }
                    l1Data.blockHashL1.set_str(blockHashString, 16);
                }

                // Parse timestamp
                if ( input["l1InfoTree"][key].contains("timestamp") &&
                     input["l1InfoTree"][key]["timestamp"].is_string() )
                {
                    string timestampString = input["l1InfoTree"][key]["timestamp"];
                    if (!stringIsDec(timestampString))
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree timestamp found in input JSON file but not a decimal string");
                        exitProcess();
                    }
                    mpz_class timestampScalar;
                    timestampScalar.set_str(timestampString, 10);
                    if (timestampScalar > ScalarMask64)
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree timestamp found in input JSON file is too big");
                        exitProcess();
                    }
                    l1Data.minTimestamp = timestampScalar.get_ui();
                }

                // Parse smtProof
                if ( input["l1InfoTree"][key].contains("smtProof") &&
                     input["l1InfoTree"][key]["smtProof"].is_array() )
                {
                    uint64_t smtProofSize = input["l1InfoTree"][key]["smtProof"].size();
                    for (uint64_t i=0; i<smtProofSize; i++)
                    {
                        string auxString = input["l1InfoTree"][key]["smtProof"][i];
                        auxString = Remove0xIfPresent(auxString);
                        if (!stringIsHex(auxString))
                        {
                            zklog.error("Input::loadGlobals() l1InfoTree smtProof found in input JSON file is not an hexa string");
                            exitProcess();
                        }
                        mpz_class auxScalar;
                        auxScalar.set_str(auxString, 16);
                        l1Data.smtProof.emplace_back(auxScalar);
                    }
                }

                l1InfoTreeData[index] = l1Data;
            }
        }
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): l1InfoTree.size=" + to_string(l1InfoTreeData.size()));
#endif
    }

    /****************/
    /* TRACE CONFIG */
    /****************/

    // Input JSON file may contain a disableStorage key at the root level
    if ( input.contains("disableStorage") &&
         input["disableStorage"].is_boolean() )
    {
        traceConfig.bEnabled = true;
        traceConfig.bDisableStorage = input["disableStorage"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() disableStorage=" + to_string(traceConfig.bDisableStorage));
#endif
    }

    // Input JSON file may contain a disableStack key at the root level
    if ( input.contains("disableStack") &&
         input["disableStack"].is_boolean() )
    {
        traceConfig.bEnabled = true;
        traceConfig.bDisableStack = input["disableStack"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() disableStack=" + to_string(traceConfig.bDisableStack));
#endif
    }

    // Input JSON file may contain a enableMemory key at the root level
    if ( input.contains("enableMemory") &&
         input["enableMemory"].is_boolean() )
    {
        traceConfig.bEnabled = true;
        traceConfig.bEnableMemory = input["enableMemory"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() enableMemory=" + to_string(traceConfig.bEnableMemory));
#endif
    }

    // Input JSON file may contain a enableReturnData key at the root level
    if ( input.contains("enableReturnData") &&
         input["enableReturnData"].is_boolean() )
    {
        traceConfig.bEnabled = true;
        traceConfig.bEnableReturnData = input["enableReturnData"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() enableReturnData=" + to_string(traceConfig.bEnableReturnData));
#endif
    }

    // Input JSON file may contain a txHashToGenerateCallTrace key at the root level
    if ( input.contains("txHashToGenerateFullTrace") &&
         input["txHashToGenerateFullTrace"].is_string() )
    {
        traceConfig.bEnabled = true;
        traceConfig.txHashToGenerateFullTrace = Add0xIfMissing(input["txHashToGenerateFullTrace"]);
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals() txHashToGenerateFullTrace=" + traceConfig.txHashToGenerateFullTrace);
#endif
    }

    // Calculate the trace configuration flags
    traceConfig.calculateFlags();
}

void Input::saveGlobals (json &input) const
{
    // Public inputs
    if (publicInputsExtended.publicInputs.forkID != 0)
    {
        input["forkID"] = publicInputsExtended.publicInputs.forkID;
    }
    if (publicInputsExtended.publicInputs.oldStateRoot != 0)
    {
        input["oldStateRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.oldStateRoot.get_str(16), 64);
    }
    if (publicInputsExtended.publicInputs.oldAccInputHash != 0)
    {
        input["oldAccInputHash"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.oldAccInputHash.get_str(16), 64);
    }
    if (publicInputsExtended.publicInputs.oldBatchNum != 0)
    {
        input["oldNumBatch"] = publicInputsExtended.publicInputs.oldBatchNum;
    }
    input["chainID"] = publicInputsExtended.publicInputs.chainID;
    if (!publicInputsExtended.publicInputs.batchL2Data.empty())
    {
        input["batchL2Data"] = Add0xIfMissing(ba2string(publicInputsExtended.publicInputs.batchL2Data));
    }
    input["sequencerAddr"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.sequencerAddr.get_str(16), 40);
    if (publicInputsExtended.publicInputs.aggregatorAddress != 0)
    {
        input["aggregatorAddress"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.aggregatorAddress.get_str(16), 40);
    }
    if (publicInputsExtended.publicInputs.forkID <= 6)
    {
        input["globalExitRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.globalExitRoot.get_str(16), 64);
        input["timestamp"] = publicInputsExtended.publicInputs.timestamp;
    }
    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        input["l1InfoRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.l1InfoRoot.get_str(16), 64);
        input["forcedBlockHashL1"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.forcedBlockHashL1.get_str(16), 64);
        input["timestampLimit"] = publicInputsExtended.publicInputs.timestampLimit;
    }
    if (!publicInputsExtended.publicInputs.witness.empty())
    {
        input["witness"] = Add0xIfMissing(ba2string(publicInputsExtended.publicInputs.witness));
    }
    if (!publicInputsExtended.publicInputs.dataStream.empty())
    {
        input["dataStream"] =  Add0xIfMissing(ba2string(publicInputsExtended.publicInputs.dataStream));
    }

    // Public inputs extended
    if (publicInputsExtended.newStateRoot != 0) input["newStateRoot"] = NormalizeTo0xNFormat(publicInputsExtended.newStateRoot.get_str(16), 64);
    if (publicInputsExtended.newAccInputHash != 0) input["newAccInputHash"] = NormalizeTo0xNFormat(publicInputsExtended.newAccInputHash.get_str(16), 64);
    if (publicInputsExtended.newLocalExitRoot != 0) input["newLocalExitRoot"] = NormalizeTo0xNFormat(publicInputsExtended.newLocalExitRoot.get_str(16), 64);
    if (publicInputsExtended.newBatchNum != 0) input["newNumBatch"] = publicInputsExtended.newBatchNum;

    // Root
    if (!from.empty() && (from != "0x")) input["from"] = from;
    input["updateMerkleTree"] = bUpdateMerkleTree;
    if (bNoCounters) input["noCounters"] = bNoCounters;
    if (bGetKeys) input["getKeys"] = bGetKeys;
    if (bSkipVerifyL1InfoRoot) input["skipVerifyL1InfoRoot"] = bSkipVerifyL1InfoRoot;
    if (bSkipFirstChangeL2Block) input["skipFirstChangeL2Block"] = bSkipFirstChangeL2Block;
    if (bSkipWriteBlockInfoRoot) input["skipWriteBlockInfoRoot"] = bSkipWriteBlockInfoRoot;
    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        if (stepsN != 0)
        {
            input["stepsN"] = stepsN;
        }
        if (debug.gasLimit != 0)
        {
            input["gasLimit"] = debug.gasLimit;
        }
        unordered_map<uint64_t, L1Data>::const_iterator it;
        for (it = l1InfoTreeData.begin(); it != l1InfoTreeData.end(); it++)
        {
            string index = to_string(it->first);
            if (it->second.globalExitRoot != 0)
            {
                input["l1InfoTree"][index]["globalExitRoot"] = NormalizeTo0xNFormat(it->second.globalExitRoot.get_str(16), 64);
            }
            if (it->second.blockHashL1 != 0)
            {
                input["l1InfoTree"][index]["blockHash"] = NormalizeTo0xNFormat(it->second.blockHashL1.get_str(16), 64);
            }
            if (it->second.minTimestamp != 0)
            {
                mpz_class auxScalar;
                auxScalar = it->second.minTimestamp;
                input["l1InfoTree"][index]["timestamp"] = auxScalar.get_str(10);
            }
            for (uint64_t i=0; i<it->second.smtProof.size(); i++)
            {
                input["l1InfoTree"][index]["smtProof"][i] = NormalizeTo0xNFormat(it->second.smtProof[i].get_str(16), 64);
            }
        }
    }

    // TraceConfig
    if (traceConfig.bEnabled)
    {
        input["disableStorage"] = traceConfig.bDisableStorage;
        input["disableStack"] = traceConfig.bDisableStack;
        input["enableMemory"] = traceConfig.bEnableMemory;
        input["enableReturnData"] = traceConfig.bEnableReturnData;
        input["txHashToGenerateFullTrace"] = traceConfig.txHashToGenerateFullTrace;
    }
}


/* Store db into database ctx.db[] */

void Input::loadDatabase (json &input)
{
    // Load witness
    if (input.contains("witness") && input["witness"].is_string())
    {
        string witness = input["witness"];
        publicInputsExtended.publicInputs.witness = string2ba(witness);
    }

    // Input JSON file must contain a db structure at the root level
    if ( !input.contains("db") ||
         !input["db"].is_structured() )
    {
#ifdef LOG_INPUT
        //zklog.info("Input::loadDatabase() warning: db key not found in input JSON file");
#endif
    }
    else
    {
#ifdef LOG_INPUT
        zklog.info("loadDatabase() db content:");
#endif
        for (json::iterator it = input["db"].begin(); it != input["db"].end(); ++it)
        {
            // Every value must be a 12-fe array if intermediate node, or 8-fe array if value
            /* Disabling DB value length since SCs are stored in DB, with any length, stored in json format
               when in debug mode, and loaded to reproduce the batch or prove from that json file
            if (!it.value().is_array() ||
                !((it.value().size()==12) || (it.value().size()==8)) )
            {
                zklog.error("Input::loadDatabase() keys value array with invalid length in input JSON file: " + it.value());
                exitProcess();
            }*/

            // Add the 16 fe elements into the database value
            vector<Goldilocks::Element> dbValue;
            for (uint64_t i=0; i<it.value().size(); i++)
            {
                Goldilocks::Element fe;
                string2fe(fr, it.value()[i], fe);
                dbValue.push_back(fe);
            }

            // Add padding zeros for value hashes to match database format, where capacity is written always
            if (dbValue.size() == 8)
            {
                dbValue.push_back(fr.zero());
                dbValue.push_back(fr.zero());
                dbValue.push_back(fr.zero());
                dbValue.push_back(fr.zero());
            }

            // Get the key fe element
            string key = Remove0xIfPresent(it.key());
            if (key.size() > 64)
            {
                zklog.error("Input::loadDatabase() found too big key size=" + to_string(key.size()));
                exitProcess();
            }
            key = NormalizeToNFormat(key, 64);

            // Add the key:value pair to the context database
            db[key] = dbValue;
#ifdef LOG_INPUT
            string s = it.value()[0];
            zklog.info("    key: " + it.key() + " value: " + s + " etc.");
#endif
        }
    }

    // Input JSON file must contain a contractsBytecode structure at the root level
    if ( !input.contains("contractsBytecode") ||
         !input["contractsBytecode"].is_structured() )
    {
#ifdef LOG_INPUT
        //zklog.info("Input::loadDatabase() warning: contractsBytecode key not found in input JSON file");
#endif
    }
    else
    {
#ifdef LOG_INPUT
        zklog.info("loadDatabase() contractsBytecode content:");
#endif
        for (json::iterator it = input["contractsBytecode"].begin(); it != input["contractsBytecode"].end(); ++it)
        {
            // Add the 16 fe elements into the database value
            vector<uint8_t> dbValue;
            string contractValue = string2ba(it.value());
            for (uint64_t i=0; i<contractValue.size(); i++)
            {
                dbValue.push_back(contractValue.at(i));
            }

            // Get the key fe element
            string key = Remove0xIfPresent(it.key());
            if (key.size() > 64)
            {
                zklog.error("Input::loadDatabase() found too big key size=" + to_string(key.size()));
                exitProcess();
            }
            key = NormalizeToNFormat(key, 64);

            // Add the key:value pair to the context database
            contractsBytecode[key] = dbValue;
#ifdef LOG_INPUT
            string s = it.value();
            zklog.info("    key: " + it.key() + " value: " + s);
#endif
        }
    }
}

void Input::db2json (json &input, const DatabaseMap::MTMap &db, string name) const
{
    input[name] = json::object();
    for(DatabaseMap::MTMap::const_iterator iter = db.begin(); iter != db.end(); iter++)
    {
        string key = NormalizeTo0xNFormat(iter->first, 64);
        vector<Goldilocks::Element> dbValue = iter->second;
        json value;
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value[i] = PrependZeros(fr.toString(dbValue[i], 16), 16);
        }
        input[name][key] = value;
    }
}

void Input::contractsBytecode2json (json &input, const DatabaseMap::ProgramMap &contractsBytecode, string name) const
{
    input[name] = json::object();
    for(DatabaseMap::ProgramMap::const_iterator iter = contractsBytecode.begin(); iter != contractsBytecode.end(); iter++)
    {
        string key = NormalizeTo0xNFormat(iter->first, 64);
        vector<uint8_t> dbValue = iter->second;
        string value = "";
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value += byte2string(dbValue[i]);
        }
        input[name][key] = "0x" + value;
    }
}

void Input::saveDatabase (json &input) const
{
    db2json(input, db, "db");
    contractsBytecode2json(input, contractsBytecode, "contractsBytecode");
}

void Input::saveDatabase (json &input, DatabaseMap &dbReadLog) const
{
    db2json(input, dbReadLog.getMTDB(), "db");
    contractsBytecode2json(input, dbReadLog.getProgramDB(), "contractsBytecode");
}
