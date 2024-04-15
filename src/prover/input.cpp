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

string Input::save (json &input) const
{
    string result = saveGlobals(input);
    result += saveDatabase(input);
    return result;
}

string Input::save (json &input, DatabaseMap &dbReadLog) const
{
    string result = saveGlobals(input);
    result += saveDatabase(input, dbReadLog);
    return result;
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

    // Parse chainID
    if ( input.contains("chainID") &&
         input["chainID"].is_number_unsigned() )
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

    if ((publicInputsExtended.publicInputs.forkID >= 7) && (publicInputsExtended.publicInputs.forkID <= 9))
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

    if ((publicInputsExtended.publicInputs.forkID >= 7) && (publicInputsExtended.publicInputs.forkID <= 9))
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
        if (config.loadDiagnosticRom)
        {
            if ( input.contains("timestamp") &&
                 input["timestamp"].is_number_unsigned() )
            {
                publicInputsExtended.publicInputs.timestamp = input["timestamp"];
            }
        }
        else
        {
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
    }

    if ((publicInputsExtended.publicInputs.forkID >= 7) && (publicInputsExtended.publicInputs.forkID <= 9))
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

    // Input JSON file may contain a newLastTimestamp key at the root level
    if ( input.contains("newLastTimestamp") &&
         input["newLastTimestamp"].is_number_unsigned() )
    {
        publicInputsExtended.newLastTimestamp = input["newLastTimestamp"];
#ifdef LOG_INPUT
        zklog.info("Input::loadGlobals(): newLastTimestamp=" + to_string(publicInputsExtended.newLastTimestamp));
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
                if ( input["l1InfoTree"][key].contains("minTimestamp") &&
                     input["l1InfoTree"][key]["minTimestamp"].is_string() )
                {
                    string minTimestampString = input["l1InfoTree"][key]["minTimestamp"];
                    if (!stringIsDec(minTimestampString))
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree minTimestamp found in input JSON file but not a decimal string");
                        exitProcess();
                    }
                    mpz_class minTimestampScalar;
                    minTimestampScalar.set_str(minTimestampString, 10);
                    if (minTimestampScalar > ScalarMask64)
                    {
                        zklog.error("Input::loadGlobals() l1InfoTree minTimestamp found in input JSON file is too big");
                        exitProcess();
                    }
                    l1Data.minTimestamp = minTimestampScalar.get_ui();
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

                // Parse smtProofPreviousIndex
                if ( input["l1InfoTree"][key].contains("smtProofPreviousIndex") &&
                     input["l1InfoTree"][key]["smtProofPreviousIndex"].is_array() )
                {
                    uint64_t smtProofPreviousIndexSize = input["l1InfoTree"][key]["smtProofPreviousIndex"].size();
                    for (uint64_t i=0; i<smtProofPreviousIndexSize; i++)
                    {
                        string auxString = input["l1InfoTree"][key]["smtProofPreviousIndex"][i];
                        auxString = Remove0xIfPresent(auxString);
                        if (!stringIsHex(auxString))
                        {
                            zklog.error("Input::loadGlobals() l1InfoTree smtProofPreviousIndex found in input JSON file is not an hexa string");
                            exitProcess();
                        }
                        mpz_class auxScalar;
                        auxScalar.set_str(auxString, 16);
                        l1Data.smtProofPreviousIndex.emplace_back(auxScalar);
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

    // fork 10 (Feijoa) input parameters

    if (publicInputsExtended.publicInputs.forkID >= 10)
    {
        // Parse old blob state root
        if ( input.contains("oldBlobStateRoot") &&
                input["oldBlobStateRoot"].is_string() )
        {
            string oldBlobStateRootString = input["oldBlobStateRoot"];
            oldBlobStateRootString = Remove0xIfPresent(oldBlobStateRootString);
            if (!stringIsHex(oldBlobStateRootString))
            {
                zklog.error("Input::loadGlobals() oldBlobStateRoot found in input JSON file but not an hex string");
                exitProcess();
            }
            if (oldBlobStateRootString.size() > 64)
            {
                zklog.error("Input::loadGlobals() oldBlobStateRoot found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.oldBlobStateRoot.set_str(oldBlobStateRootString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): oldBlobStateRoot=" + publicInputsExtended.publicInputs.oldBlobStateRoot.get_str(16));
#endif
        }

        // Parse old blob acc input hash
        if ( input.contains("oldBlobAccInputHash") &&
                input["oldBlobAccInputHash"].is_string() )
        {
            string oldBlobAccInputHashString = input["oldBlobAccInputHash"];
            oldBlobAccInputHashString = Remove0xIfPresent(oldBlobAccInputHashString);
            if (!stringIsHex(oldBlobAccInputHashString))
            {
                zklog.error("Input::loadGlobals() oldBlobAccInputHash found in input JSON file but not an hex string");
                exitProcess();
            }
            if (oldBlobAccInputHashString.size() > 64)
            {
                zklog.error("Input::loadGlobals() oldBlobAccInputHash found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.oldBlobAccInputHash.set_str(oldBlobAccInputHashString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): oldBlobAccInputHash=" + publicInputsExtended.publicInputs.oldBlobAccInputHash.get_str(16));
#endif
        }

        // Parse old blob number
        if ( input.contains("oldNumBlob") &&
            input["oldNumBlob"].is_number_unsigned() )
        {
            publicInputsExtended.publicInputs.oldBlobNum = input["oldNumBlob"];
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): oldBlobNum=" + to_string(publicInputsExtended.publicInputs.oldBlobNum));
#endif
        }

        // Parse lastL1InfoTreeIndex
        if ( input.contains("lastL1InfoTreeIndex") &&
            input["lastL1InfoTreeIndex"].is_number_unsigned() )
        {
            publicInputsExtended.publicInputs.lastL1InfoTreeIndex = input["lastL1InfoTreeIndex"];
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): lastL1InfoTreeIndex=" + to_string(lpublicInputsExtended.publicInputs.lastL1InfoTreeIndex));
#endif
        }

        // Parse lastL1InfoTreeRoot
        if ( input.contains("lastL1InfoTreeRoot") &&
                input["lastL1InfoTreeRoot"].is_string() )
        {
            string lastL1InfoTreeRootString = input["lastL1InfoTreeRoot"];
            lastL1InfoTreeRootString = Remove0xIfPresent(lastL1InfoTreeRootString);
            if (!stringIsHex(lastL1InfoTreeRootString))
            {
                zklog.error("Input::loadGlobals() lastL1InfoTreeRoot found in input JSON file but not an hex string");
                exitProcess();
            }
            if (lastL1InfoTreeRootString.size() > 64)
            {
                zklog.error("Input::loadGlobals() lastL1InfoTreeRoot found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.lastL1InfoTreeRoot.set_str(lastL1InfoTreeRootString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): lastL1InfoTreeRoot=" + publicInputsExtended.publicInputs.lastL1InfoTreeRoot.get_str(16));
#endif
        }

        // Parse zkGasLimit
        if ( input.contains("zkGasLimit") &&
             input["zkGasLimit"].is_string() )
        {
            string zkGasLimitString = input["zkGasLimit"];
            zkGasLimitString = Remove0xIfPresent(zkGasLimitString);
            if (!stringIsDec(zkGasLimitString))
            {
                zklog.error("Input::loadGlobals() zkGasLimit found in input JSON file but not a decimal string");
                exitProcess();
            }
            if (zkGasLimitString.size() > 78)
            {
                zklog.error("Input::loadGlobals() zkGasLimit found in input JSON file is too long");
                exitProcess();
            }
            mpz_class auxScalar;
            auxScalar.set_str(zkGasLimitString, 10);
            if (auxScalar > ScalarMask64)
            {
                zklog.error("Input::loadGlobals() zkGasLimit found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.zkGasLimit = auxScalar.get_ui();
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): zkGasLimit=" + to_string(publicInputsExtended.publicInputs.zkGasLimit));
#endif
        }

        // Parse pointZ
        if ( input.contains("pointZ") &&
             input["pointZ"].is_string() )
        {
            string pointZString = input["pointZ"];
            pointZString = Remove0xIfPresent(pointZString);
            if (!stringIsHex(pointZString))
            {
                zklog.error("Input::loadGlobals() pointZ found in input JSON file but not an hex string");
                exitProcess();
            }
            if (pointZString.size() > 64)
            {
                zklog.error("Input::loadGlobals() pointZ found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.pointZ.set_str(pointZString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): pointZ=" + publicInputsExtended.publicInputs.pointZ.get_str(16));
#endif
        }
        else if ( input.contains("z") &&
                  input["z"].is_string() )
        {
            string pointZString = input["z"];
            pointZString = Remove0xIfPresent(pointZString);
            if (!stringIsHex(pointZString))
            {
                zklog.error("Input::loadGlobals() pointZ found in input JSON file but not an hex string");
                exitProcess();
            }
            if (pointZString.size() > 64)
            {
                zklog.error("Input::loadGlobals() pointZ found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.pointZ.set_str(pointZString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): pointZ=" + publicInputsExtended.publicInputs.pointZ.get_str(16));
#endif
        }

        // Parse pointY
        if ( input.contains("pointY") &&
             input["pointY"].is_string() )
        {
            string pointYString = input["pointY"];
            pointYString = Remove0xIfPresent(pointYString);
            if (!stringIsHex(pointYString))
            {
                zklog.error("Input::loadGlobals() pointY found in input JSON file but not an hex string");
                exitProcess();
            }
            if (pointYString.size() > 64)
            {
                zklog.error("Input::loadGlobals() pointY found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.pointY.set_str(pointYString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): pointY=" + publicInputsExtended.publicInputs.pointY.get_str(16));
#endif
        }
        else if ( input.contains("y") &&
                  input["y"].is_string() )
        {
            string pointYString = input["y"];
            pointYString = Remove0xIfPresent(pointYString);
            if (!stringIsHex(pointYString))
            {
                zklog.error("Input::loadGlobals() pointY found in input JSON file but not an hex string");
                exitProcess();
            }
            if (pointYString.size() > 64)
            {
                zklog.error("Input::loadGlobals() pointY found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.pointY.set_str(pointYString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): pointY=" + publicInputsExtended.publicInputs.pointY.get_str(16));
#endif
        }

        // Parse blobData
        if ( input.contains("blobData") &&
            input["blobData"].is_string() )
        {
            publicInputsExtended.publicInputs.blobData = string2ba(input["blobData"]);
        }

        // Parse blobL2HashData
        if ( input.contains("blobL2HashData") &&
             input["blobL2HashData"].is_string() )
        {
            string blobL2HashDataString = input["blobL2HashData"];
            blobL2HashDataString = Remove0xIfPresent(blobL2HashDataString);
            if (!stringIsHex(blobL2HashDataString))
            {
                zklog.error("Input::loadGlobals() blobL2HashData found in input JSON file but not an hex string");
                exitProcess();
            }
            if (blobL2HashDataString.size() > 64)
            {
                zklog.error("Input::loadGlobals() blobL2HashData found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.blobL2HashData.set_str(blobL2HashDataString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): blobL2HashData=" + publicInputsExtended.publicInputs.blobL2HashData.get_str(16));
#endif
        }

        // Parse batchHashData
        if ( input.contains("batchHashData") &&
             input["batchHashData"].is_string() )
        {
            string batchHashDataString = input["batchHashData"];
            batchHashDataString = Remove0xIfPresent(batchHashDataString);
            if (!stringIsHex(batchHashDataString))
            {
                zklog.error("Input::loadGlobals() batchHashData found in input JSON file but not an hex string");
                exitProcess();
            }
            if (batchHashDataString.size() > 64)
            {
                zklog.error("Input::loadGlobals() batchHashData found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.batchHashData.set_str(batchHashDataString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): batchHashData=" + publicInputsExtended.publicInputs.batchHashData.get_str(16));
#endif
        }

        // Parse type
        if ( input.contains("blobType") &&
             input["blobType"].is_number_unsigned() )
        {
            publicInputsExtended.publicInputs.blobType = input["blobType"];
        }

        // Parse currentL1InfoTreeRoot
        if ( input.contains("currentL1InfoTreeRoot") &&
             input["currentL1InfoTreeRoot"].is_string() )
        {
            string currentL1InfoTreeRootString = input["currentL1InfoTreeRoot"];
            currentL1InfoTreeRootString = Remove0xIfPresent(currentL1InfoTreeRootString);
            if (!stringIsHex(currentL1InfoTreeRootString))
            {
                zklog.error("Input::loadGlobals() currentL1InfoTreeRoot found in input JSON file but not an hex string");
                exitProcess();
            }
            if (currentL1InfoTreeRootString.size() > 64)
            {
                zklog.error("Input::loadGlobals() currentL1InfoTreeRoot found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.currentL1InfoTreeRoot.set_str(currentL1InfoTreeRootString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): currentL1InfoTreeRoot=" + publicInputsExtended.currentL1InfoTreeRoot.get_str(16));
#endif
        }

        // Parse currentL1InfoTreeIndex
        if ( input.contains("currentL1InfoTreeIndex") &&
             input["currentL1InfoTreeIndex"].is_number_unsigned() )
        {
            publicInputsExtended.currentL1InfoTreeIndex = input["currentL1InfoTreeIndex"];
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): currentL1InfoTreeIndex=" + to_string(lpublicInputsExtended.currentL1InfoTreeIndex));
#endif
        }

        // Parse newBlobStateRoot
        if ( input.contains("newBlobStateRoot") &&
             input["newBlobStateRoot"].is_string() )
        {
            string newBlobStateRootString = input["newBlobStateRoot"];
            newBlobStateRootString = Remove0xIfPresent(newBlobStateRootString);
            if (!stringIsHex(newBlobStateRootString))
            {
                zklog.error("Input::loadGlobals() newBlobStateRoot found in input JSON file but not an hex string");
                exitProcess();
            }
            if (newBlobStateRootString.size() > 64)
            {
                zklog.error("Input::loadGlobals() newBlobStateRoot found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.newBlobStateRoot.set_str(newBlobStateRootString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): newBlobStateRoot=" + publicInputsExtended.newBlobStateRoot.get_str(16));
#endif
        }

        // Parse newBlobAccInputHash
        if ( input.contains("newBlobAccInputHash") &&
             input["newBlobAccInputHash"].is_string() )
        {
            string newBlobAccInputHashString = input["newBlobAccInputHash"];
            newBlobAccInputHashString = Remove0xIfPresent(newBlobAccInputHashString);
            if (!stringIsHex(newBlobAccInputHashString))
            {
                zklog.error("Input::loadGlobals() newBlobAccInputHash found in input JSON file but not an hex string");
                exitProcess();
            }
            if (newBlobAccInputHashString.size() > 64)
            {
                zklog.error("Input::loadGlobals() newBlobAccInputHash found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.newBlobAccInputHash.set_str(newBlobAccInputHashString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): newBlobAccInputHash=" + publicInputsExtended.newBlobAccInputHash.get_str(16));
#endif
        }

        // Parse newBlobNum
        if ( input.contains("newNumBlob") &&
             input["newNumBlob"].is_number_unsigned() )
        {
            publicInputsExtended.newBlobNum = input["newNumBlob"];
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): newBlobNum=" + to_string(lpublicInputsExtended.newBlobNum));
#endif
        }

        // Parse finalAccBatchHashData
        if ( input.contains("finalAccBatchHashData") &&
                input["finalAccBatchHashData"].is_string() )
        {
            string finalAccBatchHashDataString = input["finalAccBatchHashData"];
            finalAccBatchHashDataString = Remove0xIfPresent(finalAccBatchHashDataString);
            if (!stringIsHex(finalAccBatchHashDataString))
            {
                zklog.error("Input::loadGlobals() finalAccBatchHashData found in input JSON file but not an hex string");
                exitProcess();
            }
            if (finalAccBatchHashDataString.size() > 64)
            {
                zklog.error("Input::loadGlobals() finalAccBatchHashData found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.finalAccBatchHashData.set_str(finalAccBatchHashDataString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): finalAccBatchHashData=" + publicInputsExtended.finalAccBatchHashData.get_str(16));
#endif
        }

        // Parse localExitRootFromBlob
        if ( input.contains("localExitRootFromBlob") &&
             input["localExitRootFromBlob"].is_string() )
        {
            string localExitRootFromBlobString = input["localExitRootFromBlob"];
            localExitRootFromBlobString = Remove0xIfPresent(localExitRootFromBlobString);
            if (!stringIsHex(localExitRootFromBlobString))
            {
                zklog.error("Input::loadGlobals() localExitRootFromBlob found in input JSON file but not an hex string");
                exitProcess();
            }
            if (localExitRootFromBlobString.size() > 64)
            {
                zklog.error("Input::loadGlobals() localExitRootFromBlob found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.localExitRootFromBlob.set_str(localExitRootFromBlobString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): localExitRootFromBlob=" + publicInputsExtended.localExitRootFromBlob.get_str(16));
#endif
        }

        // Parse isInvalid
        if ( input.contains("isInvalid") &&
             input["isInvalid"].is_boolean() )
        {
            publicInputsExtended.isInvalid = input["isInvalid"];
    #ifdef LOG_INPUT
            zklog.info("Input::loadGlobals() isInvalid=" + to_string(publicInputsExtended.isInvalid));
    #endif
        }

        // Parse forcedData
        if ( input.contains("forcedData") &&
             input["forcedData"].is_object() )
        {
            // Parse forcedData.globalExitRoot
            if ( input["forcedData"].contains("globalExitRoot") &&
                 input["forcedData"]["globalExitRoot"].is_string() )
            {
                string forcedDataString = input["forcedData"]["globalExitRoot"];
                forcedDataString = Remove0xIfPresent(forcedDataString);
                if (!stringIsHex(forcedDataString))
                {
                    zklog.error("Input::loadGlobals() forcedData found in input JSON file but not an hex string");
                    exitProcess();
                }
                if (forcedDataString.size() > 64)
                {
                    zklog.error("Input::loadGlobals() forcedData found in input JSON file is too long");
                    exitProcess();
                }
                publicInputsExtended.publicInputs.forcedData.globalExitRoot.set_str(forcedDataString, 16);
#ifdef LOG_INPUT
                zklog.info("Input::loadGlobals(): forcedData.globalExitRoot=" + publicInputsExtended.publicInputs.forcedData.globalExitRoot.get_str(16));
#endif
            }

            // Parse forcedData.blockHashL1
            if ( input["forcedData"].contains("blockHashL1") &&
                 input["forcedData"]["blockHashL1"].is_string() )
            {
                string blockHashL1String = input["forcedData"]["blockHashL1"];
                blockHashL1String = Remove0xIfPresent(blockHashL1String);
                if (!stringIsHex(blockHashL1String))
                {
                    zklog.error("Input::loadGlobals() forcedData.blockHashL1 found in input JSON file but not an hex string");
                    exitProcess();
                }
                if (blockHashL1String.size() > 64)
                {
                    zklog.error("Input::loadGlobals() forcedData.blockHashL1 found in input JSON file is too long");
                    exitProcess();
                }
                publicInputsExtended.publicInputs.forcedData.blockHashL1.set_str(blockHashL1String, 16);
#ifdef LOG_INPUT
                zklog.info("Input::loadGlobals(): forcedData.blockHashL1=" + publicInputsExtended.publicInputs.blockHashL1.get_str(16));
#endif
            }

            // Parse forcedData.minTimestamp
            if ( input["forcedData"].contains("minTimestamp") &&
                 input["forcedData"]["minTimestamp"].is_string() )
            {
                string minTimestampString = input["forcedData"]["minTimestamp"];
                minTimestampString = Remove0xIfPresent(minTimestampString);
                if (stringIsDec(minTimestampString))
                {
                    mpz_class minTimestampScalar;
                    minTimestampScalar.set_str(minTimestampString, 10);
                    if (minTimestampScalar > ScalarMask64)
                    {
                        zklog.error("Input::loadGlobals() forcedData.minTimestamp key found in input JSON file is too big value=" + minTimestampString);
                        exitProcess();
                    }
                    publicInputsExtended.publicInputs.forcedData.minTimestamp = minTimestampScalar.get_ui();
                }
                else if (stringIsHex(minTimestampString))
                {
                    mpz_class minTimestampScalar;
                    minTimestampScalar.set_str(minTimestampString, 16);
                    if (minTimestampScalar > ScalarMask64)
                    {
                        zklog.error("Input::loadGlobals() forcedData.minTimestamp key found in input JSON file is too big value=" + minTimestampString);
                        exitProcess();
                    }
                    publicInputsExtended.publicInputs.forcedData.minTimestamp = minTimestampScalar.get_ui();
                }
                else
                {
                    zklog.error("Input::loadGlobals() forcedData.minTimestamp key found in input JSON file is not decimal nor hexa value=" + minTimestampString);
                    exitProcess();
                }
#ifdef LOG_INPUT
                zklog.info("Input::loadGlobals(): forcedData.minTimestamp=" + to_string(publicInputsExtended.publicInputs.forcedData.minTimestamp));
#endif

            }

        }

        // Parse forcedHashData
        if ( input.contains("forcedHashData") &&
             input["forcedHashData"].is_string() )
        {
            string forcedHashDataString = input["forcedHashData"];
            forcedHashDataString = Remove0xIfPresent(forcedHashDataString);
            if (!stringIsHex(forcedHashDataString))
            {
                zklog.error("Input::loadGlobals() forcedHashData found in input JSON file but not an hex string");
                exitProcess();
            }
            if (forcedHashDataString.size() > 64)
            {
                zklog.error("Input::loadGlobals() forcedHashData found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.forcedHashData.set_str(forcedHashDataString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): forcedHashData=" + publicInputsExtended.publicInputs.forcedHashData.get_str(16));
#endif
        }


        // Parse previousL1InfoTreeIndex
        if ( input.contains("previousL1InfoTreeIndex") &&
             input["previousL1InfoTreeIndex"].is_number_unsigned() )
        {
            publicInputsExtended.publicInputs.previousL1InfoTreeIndex = input["previousL1InfoTreeIndex"];
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): previousL1InfoTreeIndex=" + to_string(publicInputsExtended.publicInputs.previousL1InfoTreeIndex));
#endif
        }

        // Parse previousL1InfoTreeRoot
        if ( input.contains("previousL1InfoTreeRoot") &&
             input["previousL1InfoTreeRoot"].is_string() )
        {
            string previousL1InfoTreeRootString = input["previousL1InfoTreeRoot"];
            previousL1InfoTreeRootString = Remove0xIfPresent(previousL1InfoTreeRootString);
            if (!stringIsHex(previousL1InfoTreeRootString))
            {
                zklog.error("Input::loadGlobals() previousL1InfoTreeRoot found in input JSON file but not an hex string");
                exitProcess();
            }
            if (previousL1InfoTreeRootString.size() > 64)
            {
                zklog.error("Input::loadGlobals() previousL1InfoTreeRoot found in input JSON file is too long");
                exitProcess();
            }
            publicInputsExtended.publicInputs.previousL1InfoTreeRoot.set_str(previousL1InfoTreeRootString, 16);
#ifdef LOG_INPUT
            zklog.info("Input::loadGlobals(): previousL1InfoTreeRoot=" + publicInputsExtended.publicInputs.previousL1InfoTreeRoot.get_str(16));
#endif
        }        
    }
}

string Input::saveGlobals (json &input) const
{
    string s;

    // Public inputs
    if (publicInputsExtended.publicInputs.forkID != 0)
    {
        input["forkID"] = publicInputsExtended.publicInputs.forkID;
        s += "forkID=" + to_string(publicInputsExtended.publicInputs.forkID) + " ";
    }
    if (publicInputsExtended.publicInputs.oldStateRoot != 0)
    {
        input["oldStateRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.oldStateRoot.get_str(16), 64);
        s += "oldStateRoot=" + publicInputsExtended.publicInputs.oldStateRoot.get_str(16) + " ";
    }
    if (publicInputsExtended.publicInputs.oldAccInputHash != 0)
    {
        input["oldAccInputHash"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.oldAccInputHash.get_str(16), 64);
        s += "oldAccInputHash=" + publicInputsExtended.publicInputs.oldAccInputHash.get_str(16) + " ";
    }
    if (publicInputsExtended.publicInputs.oldBatchNum != 0)
    {
        input["oldNumBatch"] = publicInputsExtended.publicInputs.oldBatchNum;
        s += "oldBatchNum=" + to_string(publicInputsExtended.publicInputs.oldBatchNum) + " ";
    }
    input["chainID"] = publicInputsExtended.publicInputs.chainID;
    s += "chainID=" + to_string(publicInputsExtended.publicInputs.chainID) + " ";
    if (!publicInputsExtended.publicInputs.batchL2Data.empty())
    {
        input["batchL2Data"] = Add0xIfMissing(ba2string(publicInputsExtended.publicInputs.batchL2Data));
        const string &data = publicInputsExtended.publicInputs.batchL2Data;
        s += "batchL2Data=" + ba2string(data.substr(0, 10)) + "..." + ba2string(data.substr(zkmax(int64_t(0),int64_t(data.size())-10), data.size())) + " ";
    }
    input["sequencerAddr"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.sequencerAddr.get_str(16), 40);
    s += "sequencerAddr=" + publicInputsExtended.publicInputs.sequencerAddr.get_str(16) + " ";
    if (publicInputsExtended.publicInputs.aggregatorAddress != 0)
    {
        input["aggregatorAddress"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.aggregatorAddress.get_str(16), 40);
        s += "aggregatorAddress=" + publicInputsExtended.publicInputs.aggregatorAddress.get_str(16) + " ";
    }
    if (publicInputsExtended.publicInputs.forkID <= 6)
    {
        input["globalExitRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.globalExitRoot.get_str(16), 64);
        s += "globalExitRoot=" + publicInputsExtended.publicInputs.globalExitRoot.get_str(16) + " ";

        input["timestamp"] = publicInputsExtended.publicInputs.timestamp;
        s += "timestamp=" + to_string(publicInputsExtended.publicInputs.timestamp) + " ";
    }
    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        input["l1InfoRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.l1InfoRoot.get_str(16), 64);
        s += "l1InfoRoot=" + publicInputsExtended.publicInputs.l1InfoRoot.get_str(16) + " ";
        
        input["forcedBlockHashL1"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.forcedBlockHashL1.get_str(16), 64);
        s += "forcedBlockHashL1=" + publicInputsExtended.publicInputs.forcedBlockHashL1.get_str(16) + " ";
        
        input["timestampLimit"] = publicInputsExtended.publicInputs.timestampLimit;
        s += "timestampLimit=" + to_string(publicInputsExtended.publicInputs.timestampLimit) + " ";
    }
    if (!publicInputsExtended.publicInputs.witness.empty())
    {
        input["witness"] = Add0xIfMissing(ba2string(publicInputsExtended.publicInputs.witness));
        const string &data = publicInputsExtended.publicInputs.witness;
        s += "witness=" + ba2string(data.substr(0, 10)) + "..." + ba2string(data.substr(zkmax(int64_t(0),int64_t(data.size())-10), data.size())) + " ";
    }
    if (!publicInputsExtended.publicInputs.dataStream.empty())
    {
        input["dataStream"] =  Add0xIfMissing(ba2string(publicInputsExtended.publicInputs.dataStream));
        const string &data = publicInputsExtended.publicInputs.dataStream;
        s += "dataStream=" + ba2string(data.substr(0, 10)) + "..." + ba2string(data.substr(zkmax(int64_t(0),int64_t(data.size())-10), data.size())) + " ";
    }

    // Public inputs extended
    if (publicInputsExtended.newStateRoot != 0)
    {
        input["newStateRoot"] = NormalizeTo0xNFormat(publicInputsExtended.newStateRoot.get_str(16), 64);
        s += "newStateRoot=" + publicInputsExtended.newStateRoot.get_str(16) + " ";
    }
    if (publicInputsExtended.newAccInputHash != 0)
    {
        input["newAccInputHash"] = NormalizeTo0xNFormat(publicInputsExtended.newAccInputHash.get_str(16), 64);
        s += "newAccInputHash=" + publicInputsExtended.newAccInputHash.get_str(16) + " ";
    }
    if (publicInputsExtended.newLocalExitRoot != 0)
    {
        input["newLocalExitRoot"] = NormalizeTo0xNFormat(publicInputsExtended.newLocalExitRoot.get_str(16), 64);
        s += "newLocalExitRoot=" + publicInputsExtended.newLocalExitRoot.get_str(16) + " ";
    }
    if (publicInputsExtended.newBatchNum != 0)
    {
        input["newNumBatch"] = publicInputsExtended.newBatchNum;
        s += "newBatchNum=" + to_string(publicInputsExtended.newBatchNum) + " ";
    }
    if (publicInputsExtended.newLastTimestamp != 0)
    {
        input["newLastTimestamp"] = publicInputsExtended.newLastTimestamp;
        s += "newLastTimestamp=" + to_string(publicInputsExtended.newLastTimestamp) + " ";
    }

    // Root
    if (!from.empty() && (from != "0x"))
    {
        input["from"] = from;
        s += "from=" + from + " ";
    }
    input["updateMerkleTree"] = bUpdateMerkleTree;
    s += "updateMerkleTree=" + to_string(bUpdateMerkleTree) + " ";
    if (bNoCounters)
    {
        input["noCounters"] = bNoCounters;
        s += "noCounters=" + to_string(bNoCounters) + " ";
    }
    if (bGetKeys)
    {
        input["getKeys"] = bGetKeys;
        s += "getKeys=" + to_string(bGetKeys) + " ";
    }
    if (bSkipVerifyL1InfoRoot)
    {
        input["skipVerifyL1InfoRoot"] = bSkipVerifyL1InfoRoot;
        s += "skipVerifyL1InfoRoot=" + to_string(bSkipVerifyL1InfoRoot) + " ";
    }
    if (bSkipFirstChangeL2Block)
    {
        input["skipFirstChangeL2Block"] = bSkipFirstChangeL2Block;
        s += "skipFirstChangeL2Block=" + to_string(bSkipFirstChangeL2Block) + " ";
    }
    if (bSkipWriteBlockInfoRoot)
    {
        input["skipWriteBlockInfoRoot"] = bSkipWriteBlockInfoRoot;
        s += "skipWriteBlockInfoRoot=" + to_string(bSkipWriteBlockInfoRoot) + " ";
    }
    if (publicInputsExtended.publicInputs.forkID >= 7)
    {
        if (stepsN != 0)
        {
            input["stepsN"] = stepsN;
            s += "stepsN=" + to_string(stepsN) + " ";
        }
        if (debug.gasLimit != 0)
        {
            input["gasLimit"] = debug.gasLimit;
            s += "gasLimit=" + to_string(debug.gasLimit) + " ";
        }
        if (!l1InfoTreeData.empty())
        {
            s += "l1InfoTreeData.size=" + to_string(l1InfoTreeData.size()) + " ";
        }
        unordered_map<uint64_t, L1Data>::const_iterator it;
        for (it = l1InfoTreeData.begin(); it != l1InfoTreeData.end(); it++)
        {
            string index = to_string(it->first);
            if (it->second.globalExitRoot != 0)
            {
                input["l1InfoTree"][index]["globalExitRoot"] = NormalizeTo0xNFormat(it->second.globalExitRoot.get_str(16), 64);
                s += "liInfoTree[" + index + "].globalExitRoot=" + it->second.globalExitRoot.get_str(16) + " ";
            }
            if (it->second.blockHashL1 != 0)
            {
                input["l1InfoTree"][index]["blockHash"] = NormalizeTo0xNFormat(it->second.blockHashL1.get_str(16), 64);
                s += "liInfoTree[" + index + "].blockHashL1=" + it->second.blockHashL1.get_str(16) + " ";
            }
            if (it->second.minTimestamp != 0)
            {
                mpz_class auxScalar;
                auxScalar = it->second.minTimestamp;
                input["l1InfoTree"][index]["minTimestamp"] = auxScalar.get_str(10);
                s += "liInfoTree[" + index + "].minTimestamp=" + auxScalar.get_str(16) + " ";
            }
            for (uint64_t i=0; i<it->second.smtProof.size(); i++)
            {
                input["l1InfoTree"][index]["smtProof"][i] = NormalizeTo0xNFormat(it->second.smtProof[i].get_str(16), 64);
            }
            for (uint64_t i=0; i<it->second.smtProofPreviousIndex.size(); i++)
            {
                input["l1InfoTree"][index]["smtProofPreviousIndex"][i] = NormalizeTo0xNFormat(it->second.smtProofPreviousIndex[i].get_str(16), 64);
            }
        }
    }

    // TraceConfig
    if (traceConfig.bEnabled)
    {
        input["disableStorage"] = traceConfig.bDisableStorage;
        s += "traceConfig.bDisableStorage=" + to_string(traceConfig.bDisableStorage) + " ";

        input["disableStack"] = traceConfig.bDisableStack;
        s += "traceConfig.bDisableStack=" + to_string(traceConfig.bDisableStack) + " ";

        input["enableMemory"] = traceConfig.bEnableMemory;
        s += "traceConfig.bEnableMemory=" + to_string(traceConfig.bEnableMemory) + " ";

        input["enableReturnData"] = traceConfig.bEnableReturnData;
        s += "traceConfig.bEnableReturnData=" + to_string(traceConfig.bEnableReturnData) + " ";

        input["txHashToGenerateFullTrace"] = traceConfig.txHashToGenerateFullTrace;
        s += "traceConfig.txHashToGenerateFullTrace=" + traceConfig.txHashToGenerateFullTrace + " ";
    }

    if (publicInputsExtended.publicInputs.forkID >= 10)
    {
        if (publicInputsExtended.publicInputs.oldBlobStateRoot != 0)
        {
            input["oldBlobStateRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.oldBlobStateRoot.get_str(16), 64);
            s += "oldBlobStateRoot=" + publicInputsExtended.publicInputs.oldBlobStateRoot.get_str(16) + " ";
        }
        if (publicInputsExtended.publicInputs.oldBlobAccInputHash != 0)
        {
            input["oldBlobAccInputHash"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.oldBlobAccInputHash.get_str(16), 64);
            s += "oldBlobAccInputHash=" + publicInputsExtended.publicInputs.oldBlobAccInputHash.get_str(16) + " ";
        }
        if (publicInputsExtended.publicInputs.oldBlobNum != 0)
        {
            input["oldNumBlob"] = publicInputsExtended.publicInputs.oldBlobNum;
            s += "oldBlobNum=" + to_string(publicInputsExtended.publicInputs.oldBlobNum) + " ";
        }
        if (publicInputsExtended.publicInputs.lastL1InfoTreeIndex != 0)
        {
            input["lastL1InfoTreeIndex"] = publicInputsExtended.publicInputs.lastL1InfoTreeIndex;
            s += "lastL1InfoTreeIndex=" + to_string(publicInputsExtended.publicInputs.lastL1InfoTreeIndex) + " ";
        }
        if (publicInputsExtended.publicInputs.lastL1InfoTreeRoot != 0)
        {
            input["lastL1InfoTreeRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.lastL1InfoTreeRoot.get_str(16), 64);
            s += "lastL1InfoTreeRoot=" + publicInputsExtended.publicInputs.lastL1InfoTreeRoot.get_str(16) + " ";
        }
        if (publicInputsExtended.publicInputs.zkGasLimit != 0)
        {
            input["zkGasLimit"] = to_string(publicInputsExtended.publicInputs.zkGasLimit);
            s += "zkGasLimit=" + to_string(publicInputsExtended.publicInputs.zkGasLimit) + " ";
        }
        if (publicInputsExtended.publicInputs.pointZ != 0)
        {
            input["pointZ"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.pointZ.get_str(16), 64);
            s += "pointZ=" + publicInputsExtended.publicInputs.pointZ.get_str(16) + " ";
        }
        if (publicInputsExtended.publicInputs.pointY != 0)
        {
            input["pointY"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.pointY.get_str(16), 64);
            s += "pointY=" + publicInputsExtended.publicInputs.pointY.get_str(16) + " ";
        }
        if (!publicInputsExtended.publicInputs.blobData.empty())
        {
            input["blobData"] = Add0xIfMissing(ba2string(publicInputsExtended.publicInputs.blobData));
            const string &data = publicInputsExtended.publicInputs.blobData;
            s += "blobData=" + ba2string(data.substr(0, 10)) + "..." + ba2string(data.substr(zkmax(int64_t(0),int64_t(data.size())-10), data.size())) + " ";
        }
        if (publicInputsExtended.publicInputs.blobL2HashData != 0)
        {
            input["blobL2HashData"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.blobL2HashData.get_str(16), 64);
            s += "blobL2HashData=" + publicInputsExtended.publicInputs.blobL2HashData.get_str(16) + " ";
        }
        if (publicInputsExtended.publicInputs.batchHashData != 0)
        {
            input["batchHashData"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.batchHashData.get_str(16), 64);
            s += "batchHashData=" + publicInputsExtended.publicInputs.batchHashData.get_str(16) + " ";
        }
        if (publicInputsExtended.currentL1InfoTreeRoot != 0)
        {
            input["currentL1InfoTreeRoot"] = NormalizeTo0xNFormat(publicInputsExtended.currentL1InfoTreeRoot.get_str(16), 64);
            s += "currentL1InfoTreeRoot=" + publicInputsExtended.currentL1InfoTreeRoot.get_str(16) + " ";
        }
        if (publicInputsExtended.currentL1InfoTreeIndex != 0)
        {
            input["currentL1InfoTreeIndex"] = publicInputsExtended.currentL1InfoTreeIndex;
            s += "currentL1InfoTreeIndex=" + to_string(publicInputsExtended.currentL1InfoTreeIndex) + " ";
        }
        if (publicInputsExtended.newBlobStateRoot != 0)
        {
            input["newBlobStateRoot"] = NormalizeTo0xNFormat(publicInputsExtended.newBlobStateRoot.get_str(16), 64);
            s += "newBlobStateRoot=" + publicInputsExtended.newBlobStateRoot.get_str(16) + " ";
        }
        if (publicInputsExtended.newBlobAccInputHash != 0)
        {
            input["newBlobAccInputHash"] = NormalizeTo0xNFormat(publicInputsExtended.newBlobAccInputHash.get_str(16), 64);
            s += "newBlobAccInputHash=" + publicInputsExtended.newBlobAccInputHash.get_str(16) + " ";
        }
        if (publicInputsExtended.newBlobNum != 0)
        {
            input["newBlobNum"] = publicInputsExtended.newBlobNum;
            s += "newBlobNum=" + to_string(publicInputsExtended.newBlobNum) + " ";
        }
        if (publicInputsExtended.finalAccBatchHashData != 0)
        {
            input["finalAccBatchHashData"] = NormalizeTo0xNFormat(publicInputsExtended.finalAccBatchHashData.get_str(16), 64);
            s += "finalAccBatchHashData=" + publicInputsExtended.finalAccBatchHashData.get_str(16) + " ";
        }
        if (publicInputsExtended.localExitRootFromBlob != 0)
        {
            input["localExitRootFromBlob"] = NormalizeTo0xNFormat(publicInputsExtended.localExitRootFromBlob.get_str(16), 64);
            s += "localExitRootFromBlob=" + publicInputsExtended.localExitRootFromBlob.get_str(16) + " ";
        }
        if (publicInputsExtended.isInvalid)
        {
            input["isInvalid"] = publicInputsExtended.isInvalid;
            s += "isInvalid=" + to_string(publicInputsExtended.isInvalid) + " ";
        }
        if (publicInputsExtended.publicInputs.blobType != 0)
        {
            input["blobType"] = publicInputsExtended.publicInputs.blobType;
            s += "blobType=" + to_string(publicInputsExtended.publicInputs.blobType) + " ";
        }
        if ( (publicInputsExtended.publicInputs.forcedData.globalExitRoot != 0) ||
             (publicInputsExtended.publicInputs.forcedData.blockHashL1 != 0) ||
             (publicInputsExtended.publicInputs.forcedData.minTimestamp != 0) )
        {
            input["forcedData"]["globalExitRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.forcedData.globalExitRoot.get_str(16), 64);
            s += "forcedData.globalExitRoot=" + publicInputsExtended.publicInputs.forcedData.globalExitRoot.get_str(16) + " ";

            input["forcedData"]["blockHashL1"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.forcedData.blockHashL1.get_str(16), 64);
            s += "forcedData.blockHashL1=" + publicInputsExtended.publicInputs.forcedData.blockHashL1.get_str(16) + " ";

            input["forcedData"]["minTimestamp"] = publicInputsExtended.publicInputs.forcedData.minTimestamp;
            s += "forcedData.minTimestamp=" + to_string(publicInputsExtended.publicInputs.forcedData.minTimestamp) + " ";
        }
        if (publicInputsExtended.publicInputs.forcedHashData != 0)
        {
            input["forcedHashData"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.forcedHashData.get_str(16), 64);
            s += "forcedHashData=" + publicInputsExtended.publicInputs.forcedHashData.get_str(16) + " ";
        }
        if (publicInputsExtended.publicInputs.previousL1InfoTreeIndex != 0)
        {
            input["previousL1InfoTreeIndex"] = publicInputsExtended.publicInputs.previousL1InfoTreeIndex;
            s += "previousL1InfoTreeIndex=" + to_string(publicInputsExtended.publicInputs.previousL1InfoTreeIndex) + " ";
        }
        if (publicInputsExtended.publicInputs.previousL1InfoTreeRoot != 0)
        {
            input["previousL1InfoTreeRoot"] = NormalizeTo0xNFormat(publicInputsExtended.publicInputs.previousL1InfoTreeRoot.get_str(16), 64);
            s += "previousL1InfoTreeRoot=" + publicInputsExtended.publicInputs.previousL1InfoTreeRoot.get_str(16) + " ";
        }
    }

    return s;
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

string Input::saveDatabase (json &input) const
{
    db2json(input, db, "db");
    contractsBytecode2json(input, contractsBytecode, "contractsBytecode");
    string s;
    if (!db.empty())
    {
        s += "db.size=" + to_string(db.size()) + " ";
    }
    if (!contractsBytecode.empty())
    {
        s += "contracts.size=" + to_string(contractsBytecode.size()) + " ";
    }
    return s;
}

string Input::saveDatabase (json &input, DatabaseMap &dbReadLog) const
{
    db2json(input, dbReadLog.getMTDB(), "db");
    contractsBytecode2json(input, dbReadLog.getProgramDB(), "contractsBytecode");
    string s;
    if (!db.empty())
    {
        s += "db.size=" + to_string(db.size()) + " ";
    }
    if (!contractsBytecode.empty())
    {
        s += "contracts.size=" + to_string(contractsBytecode.size()) + " ";
    }
    return s;
}

