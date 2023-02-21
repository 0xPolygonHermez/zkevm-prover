#include <iostream>
#include "config.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "database.hpp"
#include "utils.hpp"

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

    // PUBLIC INPUTS

    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("oldStateRoot") ||
         !input["oldStateRoot"].is_string() )
    {
        cerr << "Error: oldStateRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    auxString = Remove0xIfPresent(input["oldStateRoot"]);
    publicInputsExtended.publicInputs.oldStateRoot.set_str(auxString, 16);
#ifdef LOG_INPUT
    cout << "loadGlobals(): oldStateRoot=" << publicInputsExtended.publicInputs.oldStateRoot.get_str(16) << endl;
#endif

    // Input JSON file may contain a oldAccInputHash key at the root level
    if ( input.contains("oldAccInputHash") &&
         input["oldAccInputHash"].is_string() )
    {
        auxString = Remove0xIfPresent(input["oldAccInputHash"]);
        publicInputsExtended.publicInputs.oldAccInputHash.set_str(auxString, 16);
#ifdef LOG_INPUT
        cout << "loadGlobals(): oldAccInputHash=" << publicInputsExtended.publicInputs.oldAccInputHash.get_str(16) << endl;
#endif
    }

    // Input JSON file must contain a oldBatchNum key at the root level
    if ( !input.contains("oldNumBatch") ||
         !input["oldNumBatch"].is_number_unsigned() )
    {
        cerr << "Error: oldNumBatch key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.publicInputs.oldBatchNum = input["oldNumBatch"];
#ifdef LOG_INPUT
    cout << "loadGlobals(): oldBatchNum=" << publicInputsExtended.publicInputs.oldBatchNum << endl;
#endif

    // Input JSON file must contain a chainID key at the root level (it is mandatory)
    if ( !input.contains("chainID") ||
         !input["chainID"].is_number_unsigned() )
    {
        cerr << "Error: chainID key not found in input JSON file" << endl;
        exitProcess();
    }
    else
    {
        publicInputsExtended.publicInputs.chainID = input["chainID"];
    }
#ifdef LOG_INPUT
    cout << "loadGlobals(): chainID=" << publicInputsExtended.publicInputs.chainID << endl;
#endif

    // Input JSON file could contain a forkID key at the root level (not mandatory, default is 0)
    if ( input.contains("forkID") &&
         input["forkID"].is_number_unsigned() )
    {
        publicInputsExtended.publicInputs.forkID = input["forkID"];
    }
#ifdef LOG_INPUT
    cout << "loadGlobals(): forkID=" << publicInputsExtended.publicInputs.forkID << endl;
#endif

    // Input JSON file must contain a batchL2Data key at the root level
    if ( !input.contains("batchL2Data") ||
         !input["batchL2Data"].is_string() )
    {
        cerr << "Error: batchL2Data key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.publicInputs.batchL2Data = string2ba(input["batchL2Data"]);

    // Check the batchL2Data length
    if (publicInputsExtended.publicInputs.batchL2Data.size() > (MAX_BATCH_L2_DATA_SIZE))
    {
        cerr << "Error: Input::loadGlobals() found batchL2Data.size()=" << publicInputsExtended.publicInputs.batchL2Data.size() << " > MAX_BATCH_L2_DATA_SIZE=" << MAX_BATCH_L2_DATA_SIZE << endl;
        exitProcess();
    }
#ifdef LOG_INPUT
    cout << "loadGlobals(): batchL2Data=" << ba2string(publicInputsExtended.publicInputs.batchL2Data) << endl;
#endif

    // Input JSON file must contain a globalExitRoot key at the root level
    if ( !input.contains("globalExitRoot") ||
         !input["globalExitRoot"].is_string() )
    {
        cerr << "Error: globalExitRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.publicInputs.globalExitRoot.set_str(Remove0xIfPresent(input["globalExitRoot"]), 16);
#ifdef LOG_INPUT
    cout << "loadGlobals(): globalExitRoot=" << publicInputsExtended.publicInputs.globalExitRoot.get_str(16) << endl;
#endif

    // Input JSON file must contain a timestamp key at the root level
    if ( !input.contains("timestamp") ||
         !input["timestamp"].is_number_unsigned() )
    {
        cerr << "Error: timestamp key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.publicInputs.timestamp = input["timestamp"];
#ifdef LOG_INPUT
    cout << "loadGlobals(): timestamp=" << publicInputsExtended.publicInputs.timestamp << endl;
#endif

    // Input JSON file must contain a sequencerAddr key at the root level
    if ( !input.contains("sequencerAddr") ||
         !input["sequencerAddr"].is_string() )
    {
        cerr << "Error: sequencerAddr key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.publicInputs.sequencerAddr.set_str(Remove0xIfPresent(input["sequencerAddr"]), 16);
#ifdef LOG_INPUT
    cout << "loadGlobals(): sequencerAddr=" << publicInputsExtended.publicInputs.sequencerAddr.get_str(16) << endl;
#endif

    // Input JSON file may contain a aggregatorAddress key at the root level
    if ( input.contains("aggregatorAddress") &&
         input["aggregatorAddress"].is_string() )
    {
        auxString = Remove0xIfPresent(input["aggregatorAddress"]);
        publicInputsExtended.publicInputs.aggregatorAddress.set_str(auxString, 16);
#ifdef LOG_INPUT
        cout << "loadGlobals(): aggregatorAddress=" << publicInputsExtended.publicInputs.aggregatorAddress.get_str(16) << endl;
#endif
    }

    // PUBLIC INPUTS EXTENDED

    // Input JSON file must contain a newStateRoot key at the root level
    if ( !input.contains("newStateRoot") ||
         !input["newStateRoot"].is_string() )
    {
        cerr << "Error: newStateRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    auxString = Remove0xIfPresent(input["newStateRoot"]);
    publicInputsExtended.newStateRoot.set_str(auxString, 16);
#ifdef LOG_INPUT
    cout << "loadGlobals(): newStateRoot=" << publicInputsExtended.newStateRoot.get_str(16) << endl;
#endif

    // Input JSON file may contain a newAccInputHash key at the root level
    if ( input.contains("newAccInputHash") &&
         input["newAccInputHash"].is_string() )
    {
        auxString = Remove0xIfPresent(input["newAccInputHash"]);
        publicInputsExtended.newAccInputHash.set_str(auxString, 16);
#ifdef LOG_INPUT
        cout << "loadGlobals(): newAccInputHash=" << publicInputsExtended.newAccInputHash.get_str(16) << endl;
#endif
    }

    // Input JSON file must contain a newLocalExitRoot key at the root level
    if ( !input.contains("newLocalExitRoot") ||
         !input["newLocalExitRoot"].is_string() )
    {
        cerr << "Error: newLocalExitRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    auxString = Remove0xIfPresent(input["newLocalExitRoot"]);
    publicInputsExtended.newLocalExitRoot.set_str(auxString, 16);

#ifdef LOG_INPUT
    cout << "loadGlobals(): newLocalExitRoot=" << publicInputsExtended.newLocalExitRoot.get_str(16) << endl;
#endif

    // Input JSON file must contain a numBatch key at the root level
    if ( !input.contains("newNumBatch") ||
         !input["newNumBatch"].is_number_unsigned() )
    {
        cerr << "Error: newNumBatch key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.newBatchNum = input["newNumBatch"];
#ifdef LOG_INPUT
    cout << "loadGlobals(): newBatchNum=" << publicInputsExtended.newBatchNum << endl;
#endif

    // ROOT

    // Input JSON file may contain a from key at the root level
    if ( input.contains("from") &&
         input["from"].is_string() )
    {
        from = Add0xIfMissing(input["from"]);
#ifdef LOG_INPUT
        cout << "loadGlobals(): from=" << from << endl;
#endif
    }

    // Input JSON file may contain an updateMerkleTree key at the root level
    if ( input.contains("updateMerkleTree") &&
         input["updateMerkleTree"].is_boolean() )
    {
        bUpdateMerkleTree = input["updateMerkleTree"];
#ifdef LOG_INPUT
        cout << "loadGlobals(): updateMerkleTree=" << bUpdateMerkleTree << endl;
#endif
    }

    // Input JSON file may contain a noCounters key at the root level
    if ( input.contains("noCounters") &&
         input["noCounters"].is_boolean() )
    {
        bNoCounters = input["noCounters"];
#ifdef LOG_INPUT
        cout << "loadGlobals(): noCounters=" << bNoCounters << endl;
#endif
    }

    // Input JSON file may contain a disableStorage key at the root level
    if ( input.contains("disableStorage") &&
         input["disableStorage"].is_boolean() )
    {
        traceConfig.bEnabled = true;
        traceConfig.bDisableStorage = input["disableStorage"];
#ifdef LOG_INPUT
        cout << "loadGlobals(): disableStorage=" << traceConfig.bDisableStorage << endl;
#endif
    }

    // Input JSON file may contain a disableStack key at the root level
    if ( input.contains("disableStack") &&
         input["disableStack"].is_boolean() )
    {
        traceConfig.bEnabled = true;
        traceConfig.bDisableStack = input["disableStack"];
#ifdef LOG_INPUT
        cout << "loadGlobals(): disableStack=" << traceConfig.bDisableStack << endl;
#endif
    }

    // Input JSON file may contain a enableMemory key at the root level
    if ( input.contains("enableMemory") &&
         input["enableMemory"].is_boolean() )
    {
        traceConfig.bEnabled = true;
        traceConfig.bEnableMemory = input["enableMemory"];
#ifdef LOG_INPUT
        cout << "loadGlobals(): enableMemory=" << traceConfig.bEnableMemory << endl;
#endif
    }

    // Input JSON file may contain a enableReturnData key at the root level
    if ( input.contains("enableReturnData") &&
         input["enableReturnData"].is_boolean() )
    {
        traceConfig.bEnabled = true;
        traceConfig.bEnableReturnData = input["enableReturnData"];
#ifdef LOG_INPUT
        cout << "loadGlobals(): enableReturnData=" << traceConfig.bEnableReturnData << endl;
#endif
    }

    // Input JSON file may contain a txHashToGenerateExecuteTrace key at the root level
    if ( input.contains("txHashToGenerateExecuteTrace") &&
         input["txHashToGenerateExecuteTrace"].is_string() )
    {
        traceConfig.bEnabled = true;
        traceConfig.txHashToGenerateExecuteTrace = Add0xIfMissing(input["txHashToGenerateExecuteTrace"]);
#ifdef LOG_INPUT
        cout << "loadGlobals(): txHashToGenerateExecuteTrace=" << txHashToGenerateExecuteTrace << endl;
#endif
    }

    // Input JSON file may contain a txHashToGenerateCallTrace key at the root level
    if ( input.contains("txHashToGenerateCallTrace") &&
         input["txHashToGenerateCallTrace"].is_string() )
    {
        traceConfig.bEnabled = true;
        traceConfig.txHashToGenerateCallTrace = Add0xIfMissing(input["txHashToGenerateCallTrace"]);
#ifdef LOG_INPUT
        cout << "loadGlobals(): txHashToGenerateCallTrace=" << txHashToGenerateCallTrace << endl;
#endif
    }

    // Calculate the trace configuration flags
    traceConfig.calculateFlags();
}

void Input::saveGlobals (json &input) const
{
    // Public inputs
    input["oldStateRoot"] = Add0xIfMissing(publicInputsExtended.publicInputs.oldStateRoot.get_str(16));
    input["oldAccInputHash"] = Add0xIfMissing(publicInputsExtended.publicInputs.oldAccInputHash.get_str(16));
    input["oldNumBatch"] = publicInputsExtended.publicInputs.oldBatchNum;
    input["chainID"] = publicInputsExtended.publicInputs.chainID;
    input["forkID"] = publicInputsExtended.publicInputs.forkID;
    input["batchL2Data"] = Add0xIfMissing(ba2string(publicInputsExtended.publicInputs.batchL2Data));
    input["globalExitRoot"] = Add0xIfMissing(publicInputsExtended.publicInputs.globalExitRoot.get_str(16));
    input["timestamp"] = publicInputsExtended.publicInputs.timestamp;
    input["sequencerAddr"] = Add0xIfMissing(publicInputsExtended.publicInputs.sequencerAddr.get_str(16));
    input["aggregatorAddress"] = Add0xIfMissing(publicInputsExtended.publicInputs.aggregatorAddress.get_str(16));

    // Public inputs extended
    input["newStateRoot"] = Add0xIfMissing(publicInputsExtended.newStateRoot.get_str(16));
    input["newAccInputHash"] = Add0xIfMissing(publicInputsExtended.newAccInputHash.get_str(16));
    input["newLocalExitRoot"] = Add0xIfMissing(publicInputsExtended.newLocalExitRoot.get_str(16));
    input["newNumBatch"] = publicInputsExtended.newBatchNum;

    // Root
    input["from"] = from;
    input["updateMerkleTree"] = bUpdateMerkleTree;
    input["noCounters"] = bNoCounters;

    // TraceConfig
    input["disableStorage"] = traceConfig.bDisableStorage;
    input["disableStack"] = traceConfig.bDisableStack;
    input["enableMemory"] = traceConfig.bEnableMemory;
    input["enableReturnData"] = traceConfig.bEnableReturnData;
    input["txHashToGenerateExecuteTrace"] = traceConfig.txHashToGenerateExecuteTrace;
    input["txHashToGenerateCallTrace"] = traceConfig.txHashToGenerateCallTrace;
}


/* Store db into database ctx.db[] */

void Input::loadDatabase (json &input)
{
    // Input JSON file must contain a db structure at the root level
    if ( !input.contains("db") ||
         !input["db"].is_structured() )
    {
#ifdef LOG_INPUT
        //cout << "Input::loadDatabase() warning: db key not found in input JSON file" << endl;
#endif
    }
    else
    {
#ifdef LOG_INPUT
        cout << "loadDatabase() db content:" << endl;
#endif
        for (json::iterator it = input["db"].begin(); it != input["db"].end(); ++it)
        {
            // Every value must be a 12-fe array if intermediate node, or 8-fe array if value
            /* Disabling DB value length since SCs are stored in DB, with any length, stored in json format
               when in debug mode, and loaded to reproduce the batch or prove from that json file
            if (!it.value().is_array() ||
                !((it.value().size()==12) || (it.value().size()==8)) )
            {
                cerr << "Error: Input::loadDatabase() keys value array with invalid length in input JSON file: " << it.value() << endl;
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

            // Get the key fe element
            string key = NormalizeToNFormat(it.key(), 64);

            // Add the key:value pair to the context database
            db[key] = dbValue;
#ifdef LOG_INPUT
            cout << "    key: " << it.key() << " value: " << it.value()[0] << " etc." << endl;
#endif
        }
    }

    // Input JSON file must contain a contractsBytecode structure at the root level
    if ( !input.contains("contractsBytecode") ||
         !input["contractsBytecode"].is_structured() )
    {
#ifdef LOG_INPUT
        //cout << "Input::loadDatabase() warning: contractsBytecode key not found in input JSON file" << endl;
#endif
    }
    else
    {
#ifdef LOG_INPUT
        cout << "loadDatabase() contractsBytecode content:" << endl;
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
            string key = NormalizeToNFormat(it.key(), 64);

            // Add the key:value pair to the context database
            contractsBytecode[key] = dbValue;
#ifdef LOG_INPUT
            cout << "    key: " << it.key() << " value: " << it.value() << endl;
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
        input[name][key] = value;
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
