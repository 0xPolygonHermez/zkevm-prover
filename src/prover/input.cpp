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
    // PUBLIC INPUTS

    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("oldStateRoot") ||
         !input["oldStateRoot"].is_string() )
    {
        cerr << "Error: oldStateRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.publicInputs.oldStateRoot = input["oldStateRoot"];
#ifdef LOG_INPUT
    cout << "loadGlobals(): oldStateRoot=" << publicInputsExtended.publicInputs.oldStateRoot << endl;
#endif

    // Input JSON file may contain a oldAccInputHash key at the root level
    if ( input.contains("oldAccInputHash") &&
         input["oldAccInputHash"].is_string() )
    {
        publicInputsExtended.publicInputs.oldAccInputHash = Add0xIfMissing(input["oldAccInputHash"]);
#ifdef LOG_INPUT
        cout << "loadGlobals(): oldAccInputHash=" << oldAccInputHash << endl;
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

    // Input JSON file could contain a chainID key at the root level (not mandatory)
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

    // Input JSON file must contain a batchL2Data key at the root level
    if ( !input.contains("batchL2Data") ||
         !input["batchL2Data"].is_string() )
    {
        cerr << "Error: batchL2Data key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.publicInputs.batchL2Data = Add0xIfMissing(input["batchL2Data"]);

    // Check the batchL2Data length
    if (publicInputsExtended.publicInputs.batchL2Data.size() > (MAX_BATCH_L2_DATA_SIZE*2 + 2))
    {
        cerr << "Error: Input::loadGlobals() found batchL2Data.size()=" << publicInputsExtended.publicInputs.batchL2Data.size() << " > (MAX_BATCH_L2_DATA_SIZE*2+2)=" << (MAX_BATCH_L2_DATA_SIZE*2+2) << endl;
        exitProcess();
    }
#ifdef LOG_INPUT
    cout << "loadGlobals(): batchL2Data=" << batchL2Data << endl;
#endif

    // Input JSON file must contain a globalExitRoot key at the root level
    if ( !input.contains("globalExitRoot") ||
         !input["globalExitRoot"].is_string() )
    {
        cerr << "Error: globalExitRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.publicInputs.globalExitRoot = input["globalExitRoot"];
#ifdef LOG_INPUT
    cout << "loadGlobals(): globalExitRoot=" << globalExitRoot << endl;
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
    publicInputsExtended.publicInputs.sequencerAddr = input["sequencerAddr"];
#ifdef LOG_INPUT
    cout << "loadGlobals(): sequencerAddr=" << publicInputsExtended.publicInputs.sequencerAddr << endl;
#endif

    // Input JSON file may contain a aggregatorAddress key at the root level
    if ( input.contains("aggregatorAddress") &&
         input["aggregatorAddress"].is_string() )
    {
        publicInputsExtended.publicInputs.aggregatorAddress = Add0xIfMissing(input["aggregatorAddress"]);
#ifdef LOG_INPUT
        cout << "loadGlobals(): aggregatorAddress=" << publicInputsExtended.publicInputs.aggregatorAddress << endl;
#endif
    }
    else
    {
        publicInputsExtended.publicInputs.aggregatorAddress = "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"; // Default aggregator address, for testing purposes
    }

    // PUBLIC INPUTS EXTENDED

    // Input JSON file must contain a newStateRoot key at the root level
    if ( !input.contains("newStateRoot") ||
         !input["newStateRoot"].is_string() )
    {
        cerr << "Error: newStateRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.newStateRoot = input["newStateRoot"];
#ifdef LOG_INPUT
    cout << "loadGlobals(): newStateRoot=" << publicInputsExtended.newStateRoot << endl;
#endif

    // Input JSON file may contain a newAccInputHash key at the root level
    if ( input.contains("newAccInputHash") &&
         input["newAccInputHash"].is_string() )
    {
        publicInputsExtended.newAccInputHash = Add0xIfMissing(input["newAccInputHash"]);
#ifdef LOG_INPUT
        cout << "loadGlobals(): newAccInputHash=" << newAccInputHash << endl;
#endif
    }

    // Input JSON file must contain a newLocalExitRoot key at the root level
    if ( !input.contains("newLocalExitRoot") ||
         !input["newLocalExitRoot"].is_string() )
    {
        cerr << "Error: newLocalExitRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputsExtended.newLocalExitRoot = input["newLocalExitRoot"];
#ifdef LOG_INPUT
    cout << "loadGlobals(): newLocalExitRoot=" << publicInputsExtended.newLocalExitRoot << endl;
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

    // Input JSON file may contain a bUpdateMerkleTree key at the root level
    if ( input.contains("updateMerkleTree") &&
         input["updateMerkleTree"].is_boolean() )
    {
        bUpdateMerkleTree = input["updateMerkleTree"];
#ifdef LOG_INPUT
        cout << "loadGlobals(): updateMerkleTree=" << bUpdateMerkleTree << endl;
#endif
    }

    // Input JSON file may contain a bNoCounters key at the root level
    if ( input.contains("noCounters") &&
         input["noCounters"].is_boolean() )
    {
        bNoCounters = input["noCounters"];
#ifdef LOG_INPUT
        cout << "loadGlobals(): noCounters=" << bNoCounters << endl;
#endif
    }

    // Input JSON file may contain a txHashToGenerateExecuteTrace key at the root level
    if ( input.contains("txHashToGenerateExecuteTrace") &&
         input["txHashToGenerateExecuteTrace"].is_string() )
    {
        txHashToGenerateExecuteTrace = Add0xIfMissing(input["txHashToGenerateExecuteTrace"]);
#ifdef LOG_INPUT
        cout << "loadGlobals(): txHashToGenerateExecuteTrace=" << txHashToGenerateExecuteTrace << endl;
#endif
    }

    // Input JSON file may contain a txHashToGenerateCallTrace key at the root level
    if ( input.contains("txHashToGenerateCallTrace") &&
         input["txHashToGenerateCallTrace"].is_string() )
    {
        txHashToGenerateCallTrace = Add0xIfMissing(input["txHashToGenerateCallTrace"]);
#ifdef LOG_INPUT
        cout << "loadGlobals(): txHashToGenerateCallTrace=" << txHashToGenerateCallTrace << endl;
#endif
    }
}

void Input::saveGlobals (json &input) const
{
    // Public inputs
    input["oldStateRoot"] = publicInputsExtended.publicInputs.oldStateRoot;
    input["oldAccInputHash"] = publicInputsExtended.publicInputs.oldAccInputHash;
    input["oldNumBatch"] = publicInputsExtended.publicInputs.oldBatchNum;
    input["chainID"] = publicInputsExtended.publicInputs.chainID;
    input["batchL2Data"] = publicInputsExtended.publicInputs.batchL2Data;
    input["globalExitRoot"] = publicInputsExtended.publicInputs.globalExitRoot;
    input["timestamp"] = publicInputsExtended.publicInputs.timestamp;
    input["sequencerAddr"] = publicInputsExtended.publicInputs.sequencerAddr;
    input["aggregatorAddress"] = publicInputsExtended.publicInputs.aggregatorAddress;

    // Public inputs extended
    input["newStateRoot"] = publicInputsExtended.newStateRoot;
    input["newAccInputHash"] = publicInputsExtended.newAccInputHash;
    input["newLocalExitRoot"] = publicInputsExtended.newLocalExitRoot;
    input["newNumBatch"] = publicInputsExtended.newBatchNum;

    // Root
    input["from"] = from;
    input["updateMerkleTree"] = bUpdateMerkleTree;
    input["noCounters"] = bNoCounters;
    input["txHashToGenerateExecuteTrace"] = txHashToGenerateExecuteTrace;
    input["txHashToGenerateCallTrace"] = txHashToGenerateCallTrace;
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
            value[i] = NormalizeToNFormat(fr.toString(dbValue[i], 16), 16);
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
